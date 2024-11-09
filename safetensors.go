// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// Tensor is a view of a Tensor within a file.
//
// It contains references to data within the full byte-buffer
// and is thus a readable view of a single tensor.
type Tensor struct {
	Name  string
	DType DType
	Shape []uint64
	Data  []byte
}

// Validate validates the object.
func (t *Tensor) Validate() error {
	numElements := numElementsFromShape(t.Shape)
	if n := uint64(len(t.Data)); n != numElements*t.DType.WordSize() {
		return fmt.Errorf("invalid tensor: dtype=%s shape=%+v len(data)=%d", t.DType, t.Shape, n)
	}
	return nil
}

// File is a structure owning some metadata to lookup tensors on a shared
// `data` byte-buffer.
type File struct {
	Tensors  []Tensor
	Metadata map[string]string
}

// Deserialize parses a byte-buffer representing the whole safetensor file and
// returns the deserialized form (no tensor allocation).
func Deserialize(buffer []byte) (*File, error) {
	var n uint64
	var err error
	r := safeTensorsHeader{}
	n, err = r.parseHeader(buffer)
	if err != nil {
		return nil, err
	}
	f := &File{Metadata: r.metadata, Tensors: make([]Tensor, len(r.tensors))}
	data := buffer[n+8:]
	for i := range r.tensors {
		r.tensors[i].toTensor(&f.Tensors[i], data)
		if err := f.Tensors[i].Validate(); err != nil {
			return nil, err
		}
	}
	return f, nil
}

// Serialize the list of tensors to an io.Writer.
func (f *File) Serialize(w io.Writer) error {
	r := safeTensorsHeader{metadata: f.Metadata, tensors: make([]tensorInfo, len(f.Tensors))}
	var offset uint64
	for i := range r.tensors {
		if err := f.Tensors[i].Validate(); err != nil {
			return err
		}
		offset = r.tensors[i].fromTensor(&f.Tensors[i], offset)
	}
	b, err := r.MarshalJSON()
	if err != nil {
		return err
	}
	// Align.
	if n := len(b) & 7; n != 0 {
		b = append(b, []byte("       "[:8-n])...)
	}
	var nbArr [8]byte
	binary.LittleEndian.PutUint64(nbArr[:], uint64(len(b)))
	if _, err := w.Write(nbArr[:]); err != nil {
		return err
	}
	if _, err := w.Write(b); err != nil {
		return err
	}
	for _, t := range f.Tensors {
		// TODO: It's unhealthy to not align the data at 8 bytes.
		if _, err := w.Write(t.Data); err != nil {
			return err
		}
	}
	return nil
}

//

// safeTensorsHeader represents the header of safetensors file.
type safeTensorsHeader struct {
	tensors  []tensorInfo
	metadata map[string]string
}

// UnmarshalJSON implements json.Unmarshaler.
//
// It keeps ordering.
func (h *safeTensorsHeader) UnmarshalJSON(data []byte) error {
	// Parse the raw JSON to maintain order
	dec := json.NewDecoder(bytes.NewReader(data))
	// Read opening brace.
	if d, err := dec.Token(); err != nil || d != json.Delim('{') {
		return err
	}
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			return err
		}
		keyStr, ok := key.(string)
		if !ok {
			return errors.New("invalid json")
		}
		if keyStr == "__metadata__" {
			if err := dec.Decode(&h.metadata); err != nil {
				return err
			}
			continue
		}
		t := tensorInfo{name: keyStr}
		if err := dec.Decode(&t); err != nil {
			return err
		}
		h.tensors = append(h.tensors, t)
	}
	if len(h.tensors) == 0 {
		return errors.New("empty tensors")
	}
	return nil
}

// MarshalJSON implements json.Marshaler.
//
// It keeps ordering.
func (h *safeTensorsHeader) MarshalJSON() ([]byte, error) {
	pairs := make([][]byte, 0, len(h.tensors)+1)
	if len(h.metadata) != 0 {
		d, err := json.Marshal(h.metadata)
		if err != nil {
			return nil, err
		}
		pairs = append(pairs, append([]byte("\"__metadata__\":"), d...))
	}
	for _, t := range h.tensors {
		k, err := json.Marshal(t.name)
		if err != nil {
			return nil, err
		}
		k = append(k, ':')
		d, err := json.Marshal(t)
		if err != nil {
			return nil, err
		}

		pairs = append(pairs, append(k, d...))
	}
	buf := bytes.Buffer{}
	buf.WriteString("{")
	for i, p := range pairs {
		if i != 0 {
			buf.WriteString(",")
		}
		buf.Write(p)
	}
	buf.WriteString("}")
	return buf.Bytes(), nil
}

// parseHeader parses the header and returns the size of the header + parsed
// data, given a byte-buffer representing the whole safetensor file.
func (h *safeTensorsHeader) parseHeader(buffer []byte) (uint64, error) {
	bufferLen := uint64(len(buffer))
	if bufferLen < 8 {
		return 0, fmt.Errorf("header (%d bytes) too small", bufferLen)
	}
	n := binary.LittleEndian.Uint64(buffer)
	if n > maxHeaderSize {
		return 0, fmt.Errorf("header too large: max %d, actual %d", maxHeaderSize, n)
	}
	stop := n + 8
	if stop > bufferLen {
		return 0, fmt.Errorf("invalid header length %d", stop)
	}
	if err := json.Unmarshal(buffer[8:stop], h); err != nil {
		return 0, fmt.Errorf("invalid header deserialization: %w", err)
	}
	bufferEnd, err := h.validate()
	if err != nil {
		return 0, err
	}
	if bufferEnd+8+n != bufferLen {
		return 0, fmt.Errorf("metadata incomplete buffer")
	}
	return n, nil
}

// validate the safeTensorsHeader object.
//
// In case of success, it returns the last seen offset position, that should
// correspond to the end of the data buffer.
func (h *safeTensorsHeader) validate() (uint64, error) {
	start := uint64(0)
	for _, info := range h.tensors {
		// TODO: We should allow empty space for 8 bytes alignment.
		if info.DataOffsets[0] != start || info.DataOffsets[1] < start {
			return 0, fmt.Errorf("invalid metadata offset for tensor %q", info.name)
		}
		numElements := uint64(1)
		for _, v := range info.Shape {
			var err error
			if numElements, err = checkedMul(numElements, v); err != nil {
				return 0, fmt.Errorf("metadata validation error: failed to compute num elements from shape: %w", err)
			}
		}

		numBytes, err := checkedMul(numElements, info.DType.WordSize())
		if err != nil {
			return 0, fmt.Errorf("metadata validation error: failed to compute num bytes from num elements: %w", err)
		}
		if info.DataOffsets[1]-start != numBytes {
			return 0, fmt.Errorf("metadata validation error: info data offsets mismatch")
		}
		start = info.DataOffsets[1]
	}
	return start, nil
}

// tensorInfo provides information of a single tensor.
//
// Endianness is assumed to be little-endian. Ordering is assumed to be 'C'.
type tensorInfo struct {
	// name is the name of the tensor. It is not part of the encoded JSON.
	name string
	// The DType of each element of the tensor.
	DType DType `json:"dtype"`
	// The Shape of the tensor.
	Shape []uint64 `json:"shape"`
	// DataOffsets provides the offsets to find the data
	// within the byte-buffer array.
	DataOffsets [2]uint64 `json:"data_offsets"`
}

func (t *tensorInfo) toTensor(dst *Tensor, data []byte) {
	dst.Name = t.name
	dst.DType = t.DType
	dst.Shape = t.Shape
	dst.Data = data[t.DataOffsets[0]:t.DataOffsets[1]]
}

func (t *tensorInfo) fromTensor(src *Tensor, offset uint64) uint64 {
	t.name = src.Name
	t.DType = src.DType
	t.Shape = src.Shape
	t.DataOffsets[0] = offset
	offset += uint64(len(src.Data))
	t.DataOffsets[1] = offset
	return offset
}

const maxHeaderSize = 100_000_000

func numElementsFromShape(shape []uint64) uint64 {
	if len(shape) == 0 {
		return 1
	}
	n := shape[0]
	for _, v := range shape[1:] {
		n *= v
	}
	return n
}

// checkedMul multiplies a and b and checks for overflow.
func checkedMul(a, b uint64) (uint64, error) {
	c := a * b
	if a > 1 && b > 1 && c/a != b {
		return c, fmt.Errorf("multiplication overflow: %d * %d", a, b)
	}
	return c, nil
}
