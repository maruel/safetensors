// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"math"

	"github.com/maruel/safetensors"
)

func ExampleDeserialize() {
	serialized := []byte("\x59\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"foo":"bar"}}` +
		"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

	loaded, err := safetensors.Deserialize(serialized)
	if err != nil {
		log.Fatal(err)
	}
	var names []string
	for _, t := range loaded.Tensors {
		names = append(names, t.Name)
	}
	fmt.Printf("len = %d\n", len(loaded.Tensors))
	fmt.Printf("names = %+v\n", names)

	tensor := loaded.Tensors[0]
	fmt.Printf("tensor type = %s\n", tensor.DType)
	fmt.Printf("tensor shape = %+v\n", tensor.Shape)
	fmt.Printf("tensor data len = %+v\n", len(tensor.Data))

	// Output:
	// len = 1
	// names = [test]
	// tensor type = I32
	// tensor shape = [2 2]
	// tensor data len = 16
}

func ExampleSerialize() {
	floatData := []float32{0, 1, 2, 3, 4, 5}
	data := make([]byte, 0, len(floatData)*4)
	for _, v := range floatData {
		data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
	}

	shape := []uint64{1, 2, 3}

	tensor := safetensors.Tensor{Name: "foo", DType: safetensors.F32, Shape: shape, Data: data}
	if err := tensor.Validate(); err != nil {
		log.Fatal(err)
	}
	st := safetensors.SafeTensors{
		Tensors: []safetensors.Tensor{tensor},
	}

	buf := bytes.Buffer{}
	if err := st.Serialize(&buf); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("data len = %d\n", buf.Len())
	fmt.Printf("data excerpt: ...%s...\n", buf.Bytes()[8:30])

	// Output:
	// data len = 96
	// data excerpt: ...{"foo":{"dtype":"F32",...
}
