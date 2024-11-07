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
		str  string
	}{
		{BOOL, 1, "BOOL"},
		{U8, 1, "U8"},
		{I8, 1, "I8"},
		{I16, 2, "I16"},
		{U16, 2, "U16"},
		{F16, 2, "F16"},
		{BF16, 2, "BF16"},
		{I32, 4, "I32"},
		{U32, 4, "U32"},
		{F32, 4, "F32"},
		{F64, 8, "F64"},
		{I64, 8, "I64"},
		{U64, 8, "U64"},
	}
	for _, tc := range data {
		if tc.in.String() != tc.str {
			t.Fatalf("%s != %s", tc.in.String(), tc.str)
		}
		if tc.in.Size() != tc.size {
			t.Fatalf("%d != %d", tc.in.Size(), tc.size)
		}
	}
}
