// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bytes"
	"encoding/binary"
	"math"
	"strconv"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestDeserialize(t *testing.T) {
	d := []byte("Y\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"foo":"bar"}}` +
		"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
	got, err := Deserialize(d)
	if err != nil {
		t.Fatal(err)
	}
	want := &SafeTensors{
		Tensors:  []Tensor{{Name: "test", DType: I32, Shape: []uint64{2, 2}, Data: make([]byte, 16)}},
		Metadata: map[string]string{"foo": "bar"},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("(-want,+got)\n%s", diff)
	}
}

func TestSerialize(t *testing.T) {
	floatData := []float32{0, 1, 2, 3, 4, 5}
	data := make([]byte, 0, len(floatData)*4)
	for _, v := range floatData {
		data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
	}
	t.Run("simple serialization", func(t *testing.T) {
		st := SafeTensors{Tensors: []Tensor{{Name: "attn.0", DType: F32, Shape: []uint64{1, 2, 3}, Data: data}}}
		buf := bytes.Buffer{}
		if err := st.Serialize(&buf); err != nil {
			t.Fatal(err)
		}
		want := []byte(
			"@\x00\x00\x00\x00\x00\x00\x00" +
				"{\"attn.0\":{\"dtype\":\"F32\",\"shape\":[1,2,3],\"data_offsets\":[0,24]}}" +
				"\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@\x00\x00\xa0@")
		if diff := cmp.Diff(want, buf.Bytes()); diff != "" {
			t.Errorf("(-want,+got)\n%s", diff)
		}
		if _, err := Deserialize(buf.Bytes()); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("forced alignment", func(t *testing.T) {
		// Smaller string to force misalignment compared to previous test.
		st := SafeTensors{Tensors: []Tensor{{Name: "attn0", DType: F32, Shape: []uint64{1, 1, 2, 3}, Data: data}}}
		buf := bytes.Buffer{}
		if err := st.Serialize(&buf); err != nil {
			t.Fatal(err)
		}
		want := []byte(
			"H\x00\x00\x00\x00\x00\x00\x00" +
				"{\"attn0\":{\"dtype\":\"F32\",\"shape\":[1,1,2,3],\"data_offsets\":[0,24]}}" +
				// All the 32 are forcing alignment of the tensor data for casting to f32, f64
				// etc..
				"       " +
				"\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@\x00\x00\xa0@")
		if diff := cmp.Diff(want, buf.Bytes()); diff != "" {
			t.Errorf("(-want,+got)\n%s", diff)
		}
		got, err := Deserialize(buf.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		wantT := &SafeTensors{
			Tensors: []Tensor{
				{
					Name:  "attn0",
					DType: F32,
					Shape: []uint64{1, 1, 2, 3},
					Data:  data,
				},
			},
		}
		if diff := cmp.Diff(wantT, got); diff != "" {
			t.Fatalf("(-want,+got)\n%s", diff)
		}
	})

	t.Run("multiple", func(t *testing.T) {
		// Make sure the deserialized version has the same order.
		st := SafeTensors{
			Tensors: []Tensor{
				{Name: "attn.0", DType: I16, Shape: []uint64{1}, Data: []byte{1, 0}},
				{Name: "attn.1", DType: I16, Shape: []uint64{2}, Data: []byte{5, 4, 3, 2}},
				{Name: "attn.2", DType: I16, Shape: []uint64{1}, Data: []byte{7, 6}},
			},
			Metadata: map[string]string{"happy": "very"},
		}
		buf := bytes.Buffer{}
		if err := st.Serialize(&buf); err != nil {
			t.Fatal(err)
		}
		want := []byte(
			"\xd0\x00\x00\x00\x00\x00\x00\x00" +
				"{\"__metadata__\":{\"happy\":\"very\"},\"attn.0\":{\"dtype\":\"I16\",\"shape\":[1],\"data_offsets\":[0,2]},\"attn.1\":{\"dtype\":\"I16\",\"shape\":[2],\"data_offsets\":[2,6]},\"attn.2\":{\"dtype\":\"I16\",\"shape\":[1],\"data_offsets\":[6,8]}}" +
				" " +
				"\x01\x00\x05\x04\x03\x02\x07\x06")
		if diff := cmp.Diff(want, buf.Bytes()); diff != "" {
			t.Errorf("(-want,+got)\n%s", diff)
		}
		got, err := Deserialize(buf.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		wantT := &SafeTensors{
			Tensors: []Tensor{
				{
					Name:  "attn.0",
					DType: I16,
					Shape: []uint64{1},
					Data:  []byte{1, 0},
				},
				{
					Name:  "attn.1",
					DType: I16,
					Shape: []uint64{2},
					Data:  []byte{5, 4, 3, 2},
				},
				{
					Name:  "attn.2",
					DType: I16,
					Shape: []uint64{1},
					Data:  []byte{7, 6},
				},
			},
			Metadata: map[string]string{"happy": "very"},
		}
		if diff := cmp.Diff(wantT, got); diff != "" {
			t.Fatalf("(-want,+got)\n%s", diff)
		}
	})
}

func TestEmptyShapesAllowed(t *testing.T) {
	d := []byte("8\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[],"data_offsets":[0,4]}}` +
		"\x01\x00\x00\x00")
	got, err := Deserialize(d)
	if err != nil {
		t.Fatal(err)
	}
	want := &SafeTensors{
		Tensors: []Tensor{{Name: "test", DType: I32, Shape: []uint64{}, Data: []byte{1, 0, 0, 0}}},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("(-want,+got)\n%s", diff)
	}
}

func TestZeroSizedTensor(t *testing.T) {
	d := []byte("<\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[2,0],"data_offsets":[0, 0]}}`)
	got, err := Deserialize(d)
	if err != nil {
		t.Fatal(err)
	}
	want := &SafeTensors{
		Tensors: []Tensor{{Name: "test", DType: I32, Shape: []uint64{2, 0}, Data: make([]byte, 0)}},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("(-want,+got)\n%s", diff)
	}
}

func TestDeserialize_Errors(t *testing.T) {
	data := []struct {
		name string
		in   []byte
		err  string
	}{
		{
			"extra data",
			[]byte("<\x00\x00\x00\x00\x00\x00\x00" +
				`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}` +
				"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00extra_bogus_data_for_polyglot_file"),
			"metadata incomplete buffer",
		},
		{
			"missing data",
			[]byte("<\x00\x00\x00\x00\x00\x00\x00" +
				`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}` +
				"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"), // <- missing 2 bytes
			"metadata incomplete buffer",
		},
		{
			"HeaderTooLarge",
			[]byte("<\x00\x00\x00\x00\xff\xff\xff" +
				`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}` +
				"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"),
			"header too large: max 100000000, actual 18446742974197923900",
		},
		{"HeaderTooSmall0", []byte{}, "header (0 bytes) too small"},
		{"HeaderTooSmall1", []byte{0}, "header (1 bytes) too small"},
		{"HeaderTooSmall2", []byte{0, 0}, "header (2 bytes) too small"},
		{"HeaderTooSmall3", []byte{0, 0, 0}, "header (3 bytes) too small"},
		{"HeaderTooSmall4", []byte{0, 0, 0, 0}, "header (4 bytes) too small"},
		{"HeaderTooSmall5", []byte{0, 0, 0, 0, 0}, "header (5 bytes) too small"},
		{"HeaderTooSmall6", []byte{0, 0, 0, 0, 0, 0}, "header (6 bytes) too small"},
		{"HeaderTooSmall7", []byte{0, 0, 0, 0, 0, 0, 0}, "header (7 bytes) too small"},
		{
			"InvalidHeaderLength",
			[]byte("<\x00\x00\x00\x00\x00\x00\x00"),
			"invalid header length 68",
		},
		{
			"InvalidHeaderNonUTF8",
			[]byte("\x01\x00\x00\x00\x00\x00\x00\x00\xff"),
			"invalid header deserialization: invalid character 'ÿ' looking for beginning of value",
		},
		{
			"InvalidHeaderNotJSON",
			[]byte("\x01\x00\x00\x00\x00\x00\x00\x00{"),
			"invalid header deserialization: unexpected end of JSON input",
		},
		{
			"InvalidInfo",
			[]byte("<\x00\x00\x00\x00\x00\x00\x00" +
				`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0, 4]}}`),
			"metadata validation error: info data offsets mismatch",
		},
		{
			// max uint64 = 18_446_744_073_709_551_615
			"overflow num elements * total shape",
			[]byte("N\x00\x00\x00\x00\x00\x00\x00" +
				`{"test":{"dtype":"I32","shape":[2,9223372036854775807],"data_offsets":[0,16]}}` +
				"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"),
			"metadata validation error: failed to compute num bytes from num elements: multiplication overflow: 18446744073709551614 * 4",
		},
		{
			"overflow the shape calculation",
			[]byte("O\x00\x00\x00\x00\x00\x00\x00" +
				`{"test":{"dtype":"I32","shape":[2,18446744073709551614],"data_offsets":[0,16]}}` +
				"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"),
			"metadata validation error: failed to compute num elements from shape: multiplication overflow: 2 * 18446744073709551614",
		},
		{
			"invalid data_offset",
			[]byte("\x75\x00\x00\x00\x00\x00\x00\x00" +
				`{"test1":{"dtype":"I32","shape":[1],"data_offsets":[0, 4]},` +
				`"test2":{"dtype":"I32","shape":[1],"data_offsets":[0, 4]}}`),
			"invalid metadata offset for tensor \"test2\"",
		},
	}
	for i, line := range data {
		t.Run(strconv.Itoa(i)+": "+line.name, func(t *testing.T) {
			if _, err := Deserialize(line.in); err == nil || err.Error() != line.err {
				t.Fatal(err)
			}
		})
	}
}

func Test_CheckedMul(t *testing.T) {
	const max = math.MaxUint64

	t.Run("no overflow", func(t *testing.T) {
		testCases := [][2]uint64{
			{0, 0},
			{0, 1},
			{0, 2},
			{1, 1},
			{1, 2},
			{max, 0},
			{max, 1},
			{max / 2, 2},
		}
		for _, tc := range testCases {
			for _, pair := range [][2]uint64{tc, {tc[1], tc[0]}} {
				want := pair[0] * pair[1]

				c, err := checkedMul(pair[0], pair[1])
				if c != want || err != nil {
					t.Errorf("%d * %d: want (%d, nil), got (%d, %v)", pair[0], pair[1], want, c, err)
				}
			}
		}
	})

	t.Run("overflow", func(t *testing.T) {
		testCases := [][2]uint64{
			{max, 2},
			{max / 2, 3},
			{max, max},
		}
		for _, tc := range testCases {
			for _, pair := range [][2]uint64{tc, {tc[1], tc[0]}} {
				c, err := checkedMul(pair[0], pair[1])
				if err == nil {
					t.Errorf("%d * %d: want error, got (%d, nil)", pair[0], pair[1], c)
				}
			}
		}
	})
}
