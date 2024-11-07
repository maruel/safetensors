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
	Metadata Metadata
	data     []byte
}

// Deserialize parses a byte-buffer representing the whole
// safetensor file and returns the deserialized form (no tensor allocation).
func Deserialize(buffer []byte) (SafeTensors, error) {
	n, metadata, err := ReadMetadata(buffer)
	if err != nil {
		return SafeTensors{}, err
	}
	return SafeTensors{
		Metadata: metadata,
		data:     buffer[n+8:],
	}, nil
}

// ReadMetadata parses the header and returns the size of the header + parsed
// data, given a byte-buffer representing the whole safetensor file.
func ReadMetadata(buffer []byte) (uint64, Metadata, error) {
	bufferLen := uint64(len(buffer))
	if bufferLen < 8 {
		return 0, Metadata{}, fmt.Errorf("header too small")
	}

	arr := buffer[:8]
	n := binary.LittleEndian.Uint64(arr)
	if n > maxHeaderSize {
		return 0, Metadata{}, fmt.Errorf("header too large: max %d, actual %d", maxHeaderSize, n)
	}

	stop := n + 8
	if stop > bufferLen {
		return 0, Metadata{}, fmt.Errorf("invalid header length")
	}

	var metadata Metadata
	err := json.Unmarshal(buffer[8:stop], &metadata)
	if err != nil {
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

// Tensors returns a list of named views of all tensors.
func (st SafeTensors) Tensors() []NamedTensorView {
	tensors := make([]NamedTensorView, len(st.Metadata.Names))
	for index, name := range st.Metadata.Names {
		info := &st.Metadata.Tensors[index]
		tensors[index] = NamedTensorView{
			Name: name,
			TensorView: TensorView{
				DType: info.DType,
				Shape: info.Shape,
				Data:  st.data[info.DataOffsets[0]:info.DataOffsets[1]],
			},
		}
	}
	return tensors
}

// Tensor retrieves a the view of a specific tensor by name.
//
// The returned boolean flag reports whether the tensor was found.
func (st SafeTensors) Tensor(name string) (TensorView, bool) {
	// Linear search for now. Normally the number of tensors is at most in the
	// low hundreds so it's not a big deal. If it becomes an issue, uses a
	// private map.
	for i, n := range st.Metadata.Names {
		if n == name {
			return TensorView{
				DType: st.Metadata.Tensors[i].DType,
				Shape: st.Metadata.Tensors[i].Shape,
				Data:  st.data[st.Metadata.Tensors[i].DataOffsets[0]:st.Metadata.Tensors[i].DataOffsets[1]],
			}, true
		}
	}
	return TensorView{}, false
}

// Serialize the dictionary of tensors to a byte buffer.
func Serialize(data map[string]TensorView, dataInfo map[string]string) ([]byte, error) {
	pd, tensors, err := prepare(data, dataInfo)
	if err != nil {
		return nil, err
	}
	expectedSize := 8 + pd.n + pd.offset
	buffer := make([]byte, 0, expectedSize)
	buffer = binary.LittleEndian.AppendUint64(buffer, pd.n)
	buffer = append(buffer, pd.headerBytes...)
	for _, tensor := range tensors {
		buffer = append(buffer, tensor.Data...)
	}
	return buffer, nil
}

// SerializeToWriter the dictionary of tensors to an io.Writer (such as a file).
//
// Compared to Serialize, this procedure reduces the need to allocate the
// whole amount of memory.
func SerializeToWriter(data map[string]TensorView, dataInfo map[string]string, w io.Writer) error {
	pd, tensors, err := prepare(data, dataInfo)
	if err != nil {
		return err
	}

	var nbArr [8]byte
	nb := nbArr[:]
	binary.LittleEndian.PutUint64(nb, pd.n)

	_, err = w.Write(nb)
	if err != nil {
		return err
	}

	_, err = w.Write(pd.headerBytes)
	if err != nil {
		return err
	}

	for _, tensor := range tensors {
		_, err = w.Write(tensor.Data)
		if err != nil {
			return err
		}
	}

	return nil
}

//

type preparedData struct {
	n           uint64
	headerBytes []byte
	offset      uint64
}

func prepare(dataMap map[string]TensorView, dataInfo map[string]string) (preparedData, []TensorView, error) {
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
		return preparedData{}, nil, fmt.Errorf("failed to JSON-marshal metadata: %w", err)
	}

	// Force alignment to 8 bytes.
	extra := (8 - len(metadataBuf)%8) % 8
	if extra > 0 {
		spaces := make([]byte, extra)
		for i := range spaces {
			spaces[i] = ' '
		}
		metadataBuf = append(metadataBuf, spaces...)
	}

	pd := preparedData{
		n:           uint64(len(metadataBuf)),
		headerBytes: metadataBuf,
		offset:      offset,
	}

	return pd, tensors, nil
}
