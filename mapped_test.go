// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package safetensors

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMapped(t *testing.T) {
	n := filepath.Join(t.TempDir(), "model.safetensors")
	serialized := []byte("\x59\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"foo":"bar"}}` +
		"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
	if err := os.WriteFile(n, serialized, 0o600); err != nil {
		t.Fatal(err)
	}
	m := Mapped{}
	if err := m.Open(n); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestMapped_Error(t *testing.T) {
	n := filepath.Join(t.TempDir(), "model.safetensors")
	m := Mapped{}
	if err := m.Open(n); err == nil {
		t.Fatal("expected error")
	}
}
