// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"sort"
)

const maxHeaderSize = 100_000_000

// SafeTensors is a structure owning some metadata to lookup tensors
// on a shared `data` byte-buffer.
type SafeTensors struct {
	Metadata
	data []byte
}

// Deserialize parses a byte-buffer representing the whole safetensor file and
// returns the deserialized form (no tensor allocation).
func Deserialize(buffer []byte) (*SafeTensors, error) {
	s := &SafeTensors{}
	var n uint64
	var err error
	n, s.Metadata, err = ReadMetadata(buffer)
	if err != nil {
		return nil, err
	}
	s.data = buffer[n+8:]
	return s, nil
}

// ReadMetadata parses the header and returns the size of the header + parsed
// data, given a byte-buffer representing the whole safetensor file.
func ReadMetadata(buffer []byte) (uint64, Metadata, error) {
	bufferLen := uint64(len(buffer))
	if bufferLen < 8 {
		return 0, Metadata{}, fmt.Errorf("header (%d bytes) too small", bufferLen)
	}
	n := binary.LittleEndian.Uint64(buffer)
	if n > maxHeaderSize {
		return 0, Metadata{}, fmt.Errorf("header too large: max %d, actual %d", maxHeaderSize, n)
	}
	stop := n + 8
	if stop > bufferLen {
		return 0, Metadata{}, fmt.Errorf("invalid header length %d", stop)
	}
	var metadata Metadata
	if err := json.Unmarshal(buffer[8:stop], &metadata); err != nil {
		return 0, Metadata{}, fmt.Errorf("invalid header deserialization: %w", err)
	}
	bufferEnd, err := metadata.validate()
	if err != nil {
		return 0, Metadata{}, err
	}
	if bufferEnd+8+n != bufferLen {
		return 0, Metadata{}, fmt.Errorf("metadata incomplete buffer")
	}
	return n, metadata, nil
}

// TensorView is a view of a Tensor within a file.
//
// It contains references to data within the full byte-buffer
// and is thus a readable view of a single tensor.
type TensorView struct {
	DType DType
	Shape []uint64
	Data  []byte
}

// Validate validates the object.
func (t *TensorView) Validate() error {
	numElements := numElementsFromShape(t.Shape)
	if n := uint64(len(t.Data)); n != numElements*t.DType.Size() {
		return fmt.Errorf("invalid tensor view: dtype=%s shape=%+v len(data)=%d", t.DType, t.Shape, n)
	}
	return nil
}

// NamedTensorView is a pair of a TensorView and its name (or label, or key).
type NamedTensorView struct {
	Name       string
	TensorView TensorView
}

// NamedTensors returns a list of named views of all tensors.
func (st *SafeTensors) NamedTensors() []NamedTensorView {
	tensors := make([]NamedTensorView, len(st.Names))
	for i, t := range st.Tensors {
		tensors[i] = NamedTensorView{
			Name:       st.Names[i],
			TensorView: TensorView{DType: t.DType, Shape: t.Shape, Data: st.data[t.DataOffsets[0]:t.DataOffsets[1]]},
		}
	}
	return tensors
}

// Tensor retrieves the view of a specific tensor by name.
//
// Returns an empty invalid tensor if not found.
func (st *SafeTensors) Tensor(name string) TensorView {
	// Linear search for now. Normally the number of tensors is at most in the
	// low hundreds so it's not a big deal. If it becomes an issue, uses a
	// private map.
	for i, n := range st.Names {
		if n == name {
			t := st.Tensors[i]
			return TensorView{DType: t.DType, Shape: t.Shape, Data: st.data[t.DataOffsets[0]:t.DataOffsets[1]]}
		}
	}
	return TensorView{}
}

// Serialize the dictionary of tensors to an io.Writer.
func Serialize(data map[string]TensorView, dataInfo map[string]string, w io.Writer) error {
	header, tensors, err := prepare(data, dataInfo)
	if err != nil {
		return err
	}

	var nbArr [8]byte
	binary.LittleEndian.PutUint64(nbArr[:], uint64(len(header)))
	if _, err = w.Write(nbArr[:]); err != nil {
		return err
	}
	if _, err = w.Write(header); err != nil {
		return err
	}
	for _, tensor := range tensors {
		if _, err = w.Write(tensor.Data); err != nil {
			return err
		}
	}
	return nil
}

//

func prepare(dataMap map[string]TensorView, dataInfo map[string]string) ([]byte, []TensorView, error) {
	// Make sure we're sorting by descending dtype alignment,
	// then by name.
	data := make([]NamedTensorView, 0, len(dataMap))
	for k, v := range dataMap {
		data = append(data, NamedTensorView{Name: k, TensorView: v})
	}
	sort.Slice(data, func(i, j int) bool {
		l, r := &data[i], &data[j]
		ldt, rdt := l.TensorView.DType.Size(), r.TensorView.DType.Size()
		return ldt > rdt || (ldt == rdt && l.Name < r.Name)
	})

	offset := uint64(0)
	m := Metadata{
		Metadata: dataInfo,
		Names:    make([]string, len(data)),
		Tensors:  make([]TensorInfo, len(data)),
	}
	tensors := make([]TensorView, len(data))
	for i, v := range data {
		tensors[i] = v.TensorView
		n := uint64(len(v.TensorView.Data))
		m.Names[i] = v.Name
		m.Tensors[i] = TensorInfo{
			DType:       v.TensorView.DType,
			Shape:       v.TensorView.Shape,
			DataOffsets: [2]uint64{offset, offset + n},
		}
		offset += n
	}

	metadataBuf, err := json.Marshal(m)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to JSON-marshal metadata: %w", err)
	}

	// Force alignment to 8 bytes.
	if extra := (8 - len(metadataBuf)%8) % 8; extra > 0 {
		metadataBuf = append(metadataBuf, "       "[:extra]...)
	}
	return metadataBuf, tensors, nil
}
