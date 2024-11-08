// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"testing"
)

func TestDType(t *testing.T) {
	var data = []struct {
		in   DType
		size uint64
	}{
		{BOOL, 1},
		{U8, 1},
		{I8, 1},
		{F8_E5M2, 1},
		{F8_E4M3, 1},
		{I16, 2},
		{U16, 2},
		{F16, 2},
		{BF16, 2},
		{I32, 4},
		{U32, 4},
		{F32, 4},
		{F64, 8},
		{I64, 8},
		{U64, 8},
	}
	if len(data) != len(DTypeToWordSize) {
		t.Fatal("oops")
	}
	for _, tc := range data {
		if tc.in.WordSize() != tc.size {
			t.Fatalf("%d != %d", tc.in.WordSize(), tc.size)
		}
	}
}
