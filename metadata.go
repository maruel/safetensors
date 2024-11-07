// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
)

// Metadata represents the header of safetensor files which allow
// indexing into the raw byte-buffer array and indicates how to interpret it.
type Metadata struct {
	// Metadata is the tensors' metadata.
	Metadata map[string]string
	Names    []string
	Tensors  []TensorInfo
}

// validate the Metadata object.
//
// In case of success, it returns the last seen offset position, that should
// correspond to the end of the data buffer.
func (m Metadata) validate() (uint64, error) {
	start := uint64(0)
	for i, info := range m.Tensors {
		s := info.DataOffsets[0]
		e := info.DataOffsets[1]

		if s != start || e < s {
			tensorName := "no_tensor"
			for index, name := range m.Names {
				if index == i {
					tensorName = name
					break
				}
			}
			return 0, fmt.Errorf("invalid metadata offset for tensor %q", tensorName)
		}
		start = e

		numElements := uint64(1)
		for _, v := range info.Shape {
			var err error
			numElements, err = checkedMul(numElements, v)
			if err != nil {
				return 0, fmt.Errorf("metadata validation error: failed to compute num elements from shape: %w", err)
			}
		}

		var err error
		numBytes, err := checkedMul(numElements, info.DType.Size())
		if err != nil {
			return 0, fmt.Errorf("metadata validation error: failed to compute num bytes from num elements: %w", err)
		}
		if e-s != numBytes {
			return 0, fmt.Errorf("metadata validation error: info data offsets mismatch")
		}
	}
	return start, nil
}

func (m *Metadata) UnmarshalJSON(data []byte) error {
	var raw map[string]map[string]any
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := dec.Decode(&raw); err != nil {
		return fmt.Errorf("failed to unmarshal Metadata: %w", err)
	}

	var metadata map[string]string
	tensors := make([]namedTensorInfo, 0, len(raw))
	for k, v := range raw {
		if k == "__metadata__" {
			var err error
			if metadata, err = unmarshalMetadata(v); err != nil {
				return err
			}
		} else {
			info, err := unmarshalTensorInfo(v)
			if err != nil {
				return fmt.Errorf("failed to JSON-decode tensor %q: %w", k, err)
			}
			tensors = append(tensors, namedTensorInfo{name: k, tensorInfo: info})
		}
	}

	// We need to sort by offsets.
	// Previous versions might have a different ordering
	// than we expect (not aligned ordered, but purely name ordered,
	// or actually any order).
	sort.Slice(tensors, func(i, j int) bool {
		a := tensors[i].tensorInfo.DataOffsets
		b := tensors[j].tensorInfo.DataOffsets
		return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1])
	})

	m.Metadata = metadata
	m.Names = make([]string, len(tensors))
	m.Tensors = make([]TensorInfo, len(tensors))
	for i, v := range tensors {
		m.Names[i] = v.name
		m.Tensors[i] = v.tensorInfo
	}
	return nil
}

func (m Metadata) MarshalJSON() ([]byte, error) {
	// TODO: Keep ordering!
	obj := make(map[string]any, len(m.Names)+1)
	if len(m.Metadata) > 0 {
		obj["__metadata__"] = m.Metadata
	}
	for index, name := range m.Names {
		obj[name] = &m.Tensors[index]
	}
	return json.Marshal(obj)
}

// TensorInfo provides information of a single tensor.
//
// Endianness is assumed to be little-endian. Ordering is assumed to be 'C'.
type TensorInfo struct {
	// The DType of each element of the tensor.
	DType DType `json:"dtype"`
	// The Shape of the tensor.
	Shape []uint64 `json:"shape"`
	// DataOffsets provides the offsets to find the data
	// within the byte-buffer array.
	DataOffsets [2]uint64 `json:"data_offsets"`
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

//

func numElementsFromShape(shape []uint64) uint64 {
	if len(shape) == 0 {
		return 0
	}
	n := shape[0]
	for _, v := range shape[1:] {
		n *= v
	}
	return n
}

// namedTensorInfo is a pair of a TensorInfo and its name (or label, or key).
type namedTensorInfo struct {
	name       string
	tensorInfo TensorInfo
}

func unmarshalMetadata(value map[string]any) (map[string]string, error) {
	result := make(map[string]string, len(value))
	for k, v := range value {
		str, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("__metadata__ %q has value %#v: expected string type, actual %T", k, v, v)
		}
		result[k] = str
	}
	return result, nil
}

func unmarshalTensorInfo(m map[string]any) (TensorInfo, error) {
	if len(m) != 3 {
		return TensorInfo{}, fmt.Errorf("invalid keys: expected 3 keys (dtype, shape, data_offsets), actual %d", len(m))
	}

	dType, err := unmarshalTIDType(m)
	if err != nil {
		return TensorInfo{}, err
	}

	shape, err := unmarshalTIShape(m)
	if err != nil {
		return TensorInfo{}, err
	}

	dataOffsets, err := unmarshalTIDataOffsets(m)
	if err != nil {
		return TensorInfo{}, err
	}

	return TensorInfo{
		DType:       dType,
		Shape:       shape,
		DataOffsets: dataOffsets,
	}, nil
}

func unmarshalTIDType(m map[string]any) (DType, error) {
	v, ok := m["dtype"]
	if !ok {
		return "", fmt.Errorf(`missing "dtype"`)
	}
	s, ok := v.(string)
	if !ok || dTypeToSize[DType(s)] == 0 {
		return "", fmt.Errorf(`invalid "dtype" value: %#v of type %T`, v, v)
	}
	return DType(s), nil
}

func unmarshalTIShape(m map[string]any) ([]uint64, error) {
	v, ok := m["shape"]
	if !ok {
		return nil, fmt.Errorf(`missing "shape"`)
	}
	values, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf(`invalid "shape" value: expected array, actual %#v`, v)
	}

	shape := make([]uint64, len(values))
	for i, val := range values {
		jn, ok := val.(json.Number)
		if !ok {
			return nil, fmt.Errorf(`invalid "shape" value: expected array of natural numbers, actual %#v`, v)
		}
		n, err := strconv.ParseUint(jn.String(), 10, 64)
		if err != nil {
			return nil, fmt.Errorf(`invalid "shape" value: expected array of natural numbers, actual %#v: %w`, v, err)
		}
		shape[i] = n
	}

	return shape, nil
}

func unmarshalTIDataOffsets(m map[string]any) ([2]uint64, error) {
	v, ok := m["data_offsets"]
	if !ok {
		return [2]uint64{}, fmt.Errorf(`missing "data_offsets"`)
	}
	values, ok := v.([]any)
	if !ok {
		return [2]uint64{}, fmt.Errorf(`invalid "data_offsets" value: expected array, actual %#v`, v)
	}
	if len(values) != 2 {
		return [2]uint64{}, fmt.Errorf(`invalid "data_offsets" value: expected array of 2 elements, actual len %d: %#v`, len(values), values)
	}

	var dataOffsets [2]uint64
	for i, val := range values {
		jn, ok := val.(json.Number)
		if !ok {
			return [2]uint64{}, fmt.Errorf(`invalid "data_offsets" value: expected array of natural numbers, actual %#v`, v)
		}
		n, err := strconv.ParseUint(jn.String(), 10, 64)
		if err != nil {
			return [2]uint64{}, fmt.Errorf(`invalid "data_offsets" value: expected array of natural numbers, actual %#v: %w`, v, err)
		}
		dataOffsets[i] = n
	}

	return dataOffsets, nil
}

// checkedMul multiplies a and b and checks for overflow.
func checkedMul(a, b uint64) (uint64, error) {
	c := a * b
	if a > 1 && b > 1 && c/a != b {
		return c, fmt.Errorf("multiplication overflow: %d * %d", a, b)
	}
	return c, nil
}
