// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import "fmt"

// TensorInfo provides information of a single tensor.
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

// NamedTensorInfo is a pair of a TensorInfo and its name (or label, or key).
type NamedTensorInfo struct {
	Name       string
	TensorInfo TensorInfo
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

// NamedTensorView is a pair of a TensorView and its name (or label, or key).
type NamedTensorView struct {
	Name       string
	TensorView TensorView
}

// NewTensorView creates a new TensorView.
func NewTensorView(dType DType, shape []uint64, data []byte) (TensorView, error) {
	n := uint64(len(data))
	numElements := numElementsFromShape(shape)

	if n != numElements*dType.Size() {
		return TensorView{}, fmt.Errorf("invalid tensor view: dtype=%s shape=%+v len(data)=%d", dType, shape, n)
	}

	return TensorView{
		DType: dType,
		Shape: shape,
		Data:  data,
	}, nil
}

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
