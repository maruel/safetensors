// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package safetensors

import (
	"io"
	"os"

	"github.com/edsrzf/mmap-go"
)

// Mapped is a read-only memory mapped SafeTensors file.
//
// This is the fastest way to use a safetensors file.
type Mapped struct {
	*File
	f io.Closer
	m mmap.MMap
}

// Close releases the memory region and the file handle.
func (s *Mapped) Close() error {
	err := s.m.Unmap()
	if err2 := s.f.Close(); err == nil {
		err = err2
	}
	return err
}

// Open opens a file and memory maps it read-only.
func (s *Mapped) Open(name string) error {
	f, err := os.OpenFile(name, os.O_RDONLY, 0o600)
	if err != nil {
		return err
	}
	m, err := mmap.Map(f, mmap.RDONLY, 0)
	if err != nil {
		_ = f.Close()
		return err
	}
	s.f = f
	s.m = m
	s.File, err = Parse(m)
	if err != nil {
		_ = s.Close()
		return err
	}
	return nil
}
