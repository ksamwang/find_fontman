package main

import (
	"embed"
	"net/http"
	"path/filepath"
	"strings"
)

//go:embed web/*
var webFS embed.FS

func (a *App) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	serveEmbedded(w, "web/index.html")
}

func (a *App) handleStatic(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimPrefix(r.URL.Path, "/")
	switch filepath.Ext(name) {
	case ".js":
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
	case ".css":
		w.Header().Set("Content-Type", "text/css; charset=utf-8")
	}
	serveEmbedded(w, "web/"+name)
}

func serveEmbedded(w http.ResponseWriter, name string) {
	b, err := webFS.ReadFile(name)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	_, _ = w.Write(b)
}
