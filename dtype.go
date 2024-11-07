// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

// DType identifies a data type.
//
// It matches the DType type at
// https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs.
type DType string

const (
	// Boolan type
	BOOL DType = "BOOL"
	// Unsigned byte
	U8 DType = "U8"
	// Signed byte
	I8 DType = "I8"
	// FP8 <https://arxiv.org/pdf/2209.05433.pdf>
	F8_E5M2 DType = "F8_E5M2"
	// FP8 <https://arxiv.org/pdf/2209.05433.pdf>
	F8_E4M3 DType = "F8_E4M3"
	// Signed integer (16-bit)
	I16 DType = "I16"
	// Unsigned integer (16-bit)
	U16 DType = "U16"
	// Half-precision floating point
	F16 DType = "F16"
	// Brain floating point
	BF16 DType = "BF16"
	// Signed integer (32-bit)
	I32 DType = "I32"
	// Unsigned integer (32-bit)
	U32 DType = "U32"
	// Floating point (32-bit)
	F32 DType = "F32"
	// Floating point (64-bit)
	F64 DType = "F64"
	// Signed integer (64-bit)
	I64 DType = "I64"
	// Unsigned integer (64-bit)
	U64 DType = "U64"
)

var (
	dTypeToSize = map[DType]uint64{
		BOOL:    1,
		U8:      1,
		I8:      1,
		F8_E5M2: 1,
		F8_E4M3: 1,
		I16:     2,
		U16:     2,
		F16:     2,
		BF16:    2,
		I32:     4,
		U32:     4,
		F32:     4,
		F64:     8,
		I64:     8,
		U64:     8,
	}
)

// Size returns the size in bytes of one element of this data type.
func (dt DType) Size() uint64 {
	return dTypeToSize[dt]
}
