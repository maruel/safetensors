// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDeserialize(t *testing.T) {
	serialized := []byte("Y\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"foo":"bar"}}` +
		"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

	loaded, err := Deserialize(serialized)
	require.NoError(t, err)

	assert.Equal(t, 1, len(loaded.Tensors))
	assert.Equal(t, []string{"test"}, loaded.Names)

	tensor := loaded.Tensor("test")
	assert.Equal(t, I32, tensor.DType)
	assert.Equal(t, []uint64{2, 2}, tensor.Shape)
	assert.Equal(t, make([]byte, 16), tensor.Data)
}

func TestSerialize(t *testing.T) {
	t.Run("simple serialization", func(t *testing.T) {
		floatData := []float32{0, 1, 2, 3, 4, 5}
		data := make([]byte, 0, len(floatData)*4)
		for _, v := range floatData {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}
		attn0 := TensorView{DType: F32, Shape: []uint64{1, 2, 3}, Data: data}
		require.NoError(t, attn0.Validate())
		metadata := map[string]TensorView{"attn.0": attn0}

		buf := bytes.Buffer{}
		require.NoError(t, Serialize(metadata, nil, &buf))
		want := []byte{
			64, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 46, 48, 34, 58, 123, 34, 100,
			116, 121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34,
			58, 91, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115,
			101, 116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 0, 0, 0, 0, 0, 0, 128, 63,
			0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64,
		}
		assert.Equal(t, want, buf.Bytes())
		_, err := Deserialize(buf.Bytes())
		require.NoError(t, err)
	})

	t.Run("forced alignment", func(t *testing.T) {
		floatData := []float32{0, 1, 2, 3, 4, 5}
		data := make([]byte, 0, len(floatData)*4)
		for _, v := range floatData {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}
		attn0 := TensorView{DType: F32, Shape: []uint64{1, 1, 2, 3}, Data: data}
		require.NoError(t, attn0.Validate())
		// Smaller string to force misalignment compared to previous test.
		metadata := map[string]TensorView{"attn0": attn0}

		buf := bytes.Buffer{}
		require.NoError(t, Serialize(metadata, nil, &buf))

		want := []byte{
			72, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 48, 34, 58, 123, 34, 100, 116,
			121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58,
			91, 49, 44, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102,
			// All the 32 are forcing alignement of the tensor data for casting to f32, f64
			// etc..
			115, 101, 116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 32, 32, 32, 32, 32,
			32, 32, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0,
			160, 64,
		}
		assert.Equal(t, want, buf.Bytes())
		n, err := Deserialize(buf.Bytes())
		require.NoError(t, err)

		wantT := []NamedTensorView{
			{
				Name: "attn0",
				TensorView: TensorView{
					DType: "F32",
					Shape: []uint64{1, 1, 2, 3},
					Data:  data,
				},
			},
		}
		if diff := cmp.Diff(wantT, n.NamedTensors()); diff != "" {
			t.Fatalf("(-want,+got)\n%s", diff)
		}
	})
}

func TestGPT2Like(t *testing.T) {
	testCases := []struct {
		name   string
		nHeads int
	}{
		{"gpt2", 12},
		{"gpt2_tiny", 6},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			type tensorDescType struct {
				name  string
				shape []uint64
			}

			tensorsDesc := make([]tensorDescType, 0)
			addTensorDesc := func(name string, shape ...uint64) {
				tensorsDesc = append(tensorsDesc, tensorDescType{name: name, shape: shape})
			}

			addTensorDesc("wte", 50257, 768)
			addTensorDesc("wpe", 1024, 768)
			for i := 0; i < tc.nHeads; i++ {
				pre := fmt.Sprintf("h.%d.", i)
				addTensorDesc(pre+"ln_1.weight", 768)
				addTensorDesc(pre+"ln_1.bias", 768)
				addTensorDesc(pre+"attn.bias", 1, 1, 1024, 1024)
				addTensorDesc(pre+"attn.c_attn.weight", 768, 2304)
				addTensorDesc(pre+"attn.c_attn.bias", 2304)
				addTensorDesc(pre+"attn.c_proj.weight", 768, 768)
				addTensorDesc(pre+"attn.c_proj.bias", 768)
				addTensorDesc(pre+"ln_2.weight", 768)
				addTensorDesc(pre+"ln_2.bias", 768)
				addTensorDesc(pre+"mlp.c_fc.weight", 768, 3072)
				addTensorDesc(pre+"mlp.c_fc.bias", 3072)
				addTensorDesc(pre+"mlp.c_proj.weight", 3072, 768)
				addTensorDesc(pre+"mlp.c_proj.bias", 768)
			}
			addTensorDesc("ln_f.weight", 768)
			addTensorDesc("ln_f.bias", 768)

			dType := F32

			dataSize := uint64(0)
			for _, td := range tensorsDesc {
				dataSize += shapeProd(td.shape)
			}
			dataSize *= dType.Size()

			allData := make([]byte, dataSize)
			metadata := make(map[string]TensorView, len(tensorsDesc))
			offset := uint64(0)
			for _, td := range tensorsDesc {
				n := shapeProd(td.shape)
				buffer := allData[offset : offset+n*dType.Size()]
				tensor := TensorView{DType: dType, Shape: td.shape, Data: buffer}
				require.NoError(t, tensor.Validate())
				metadata[td.name] = tensor
				offset += n
			}

			buf := bytes.Buffer{}
			require.NoError(t, Serialize(metadata, nil, &buf))
			_, err := Deserialize(buf.Bytes())
			require.NoError(t, err)
		})
	}
}

func TestEmptyShapesAllowed(t *testing.T) {
	serialized := []byte("8\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[],"data_offsets":[0,4]}}` +
		"\x00\x00\x00\x00")

	loaded, err := Deserialize(serialized)
	require.NoError(t, err)
	assert.Equal(t, []string{"test"}, loaded.Names)
	tensor := loaded.Tensor("test")
	assert.Equal(t, I32, tensor.DType)
	assert.Equal(t, []uint64{}, tensor.Shape)
	assert.Equal(t, []byte{0, 0, 0, 0}, tensor.Data)
}

func TestJSONAttack(t *testing.T) {
	tensors := make(map[string]TensorInfo, 10)
	dType := F32
	shape := []uint64{2, 2}
	dataOffsets := [2]uint64{0, 16}

	for i := 0; i < 10; i++ {
		tensors[fmt.Sprintf("weight_%d", i)] = TensorInfo{
			DType:       dType,
			Shape:       shape,
			DataOffsets: dataOffsets,
		}
	}

	serialized, err := json.Marshal(tensors)
	require.NoError(t, err)

	n := uint64(len(serialized))

	buf := bytes.Buffer{}
	var nbArr [8]byte
	binary.LittleEndian.PutUint64(nbArr[:], n)
	_, err = buf.Write(nbArr[:])
	require.NoError(t, err)

	_, err = buf.Write(serialized)
	require.NoError(t, err)

	_, err = buf.Write([]byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	require.NoError(t, err)

	_, err = Deserialize(buf.Bytes())
	assert.ErrorContains(t, err, "invalid metadata offset for tensor")
}

func TestMetadataIncompleteBuffer(t *testing.T) {
	t.Run("extra data", func(t *testing.T) {
		serialized := []byte("<\x00\x00\x00\x00\x00\x00\x00" +
			`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}` +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00extra_bogus_data_for_polyglot_file")

		_, err := Deserialize(serialized)
		assert.EqualError(t, err, "metadata incomplete buffer")
	})

	t.Run("missing data", func(t *testing.T) {
		serialized := []byte("<\x00\x00\x00\x00\x00\x00\x00" +
			`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}` +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00") // <- missing 2 bytes

		_, err := Deserialize(serialized)
		assert.EqualError(t, err, "metadata incomplete buffer")
	})
}

func TestHeaderTooLarge(t *testing.T) {
	serialized := []byte("<\x00\x00\x00\x00\xff\xff\xff" +
		`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}` +
		"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

	_, err := Deserialize(serialized)
	assert.ErrorContains(t, err, "header too large")
}

func TestHeaderTooSmall(t *testing.T) {
	for i := range 8 {
		data := make([]byte, i)
		_, err := Deserialize(data)
		assert.EqualErrorf(t, err, fmt.Sprintf("header (%d bytes) too small", i), "data len = %d", i)
	}
}

func TestInvalidHeaderLength(t *testing.T) {
	serialized := []byte("<\x00\x00\x00\x00\x00\x00\x00")
	_, err := Deserialize(serialized)
	assert.EqualError(t, err, "invalid header length 68")
}

func TestInvalidHeaderNonUTF8(t *testing.T) {
	serialized := []byte("\x01\x00\x00\x00\x00\x00\x00\x00\xff")
	_, err := Deserialize(serialized)
	assert.ErrorContains(t, err, "invalid header deserialization")
}

func TestInvalidHeaderNotJSON(t *testing.T) {
	serialized := []byte("\x01\x00\x00\x00\x00\x00\x00\x00{")
	_, err := Deserialize(serialized)
	assert.ErrorContains(t, err, "invalid header deserialization")
}

func TestZeroSizedTensor(t *testing.T) {
	serialized := []byte("<\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[2,0],"data_offsets":[0, 0]}}`)

	loaded, err := Deserialize(serialized)
	require.NoError(t, err)
	require.Equal(t, []string{"test"}, loaded.Names)
	tensor := loaded.Tensor("test")
	assert.Equal(t, I32, tensor.DType)
	assert.Equal(t, []uint64{2, 0}, tensor.Shape)
	assert.Equal(t, []byte{}, tensor.Data)
}

func TestInvalidInfo(t *testing.T) {
	serialized := []byte("<\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0, 4]}}`)

	_, err := Deserialize(serialized)
	assert.EqualError(t, err, "metadata validation error: info data offsets mismatch")
}

func TestValidationOverflow(t *testing.T) {
	// max uint64 = 18_446_744_073_709_551_615

	t.Run("overflow the shape calculation", func(t *testing.T) {
		serialized := []byte("O\x00\x00\x00\x00\x00\x00\x00" +
			`{"test":{"dtype":"I32","shape":[2,18446744073709551614],"data_offsets":[0,16]}}` +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

		_, err := Deserialize(serialized)
		assert.ErrorContains(t, err, "metadata validation error: failed to compute num elements from shape: multiplication overflow")
	})

	t.Run("overflow num elements * total shape", func(t *testing.T) {
		serialized := []byte("N\x00\x00\x00\x00\x00\x00\x00" +
			`{"test":{"dtype":"I32","shape":[2,9223372036854775807],"data_offsets":[0,16]}}` +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

		_, err := Deserialize(serialized)
		assert.ErrorContains(t, err, "metadata validation error: failed to compute num bytes from num elements: multiplication overflow")
	})
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

//

func shapeProd(shape []uint64) uint64 {
	if len(shape) == 0 {
		return 0
	}
	p := shape[0]
	for _, v := range shape[1:] {
		p *= v
	}
	return p
}
