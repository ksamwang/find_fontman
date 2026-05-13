package main

import (
	"net/http"
	"os"
	"path/filepath"
	"sync"
)

const (
	defaultAddr       = ":8080"
	defaultVisionAddr = "127.0.0.1:9091"
)

type App struct {
	Root       string
	DataDir    string
	UploadDir  string
	PreviewDir string
	FontsDir   string

	Vision *VisionClient

	visionMu sync.Mutex
}

func NewApp() (*App, error) {
	root, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	app := &App{
		Root:       root,
		DataDir:    filepath.Join(root, "data"),
		UploadDir:  filepath.Join(root, "data", "uploads"),
		PreviewDir: filepath.Join(root, "data", "previews"),
		FontsDir:   filepath.Join(root, "fonts"),
	}
	app.Vision = NewVisionClient(app, "http://"+env("VISION_ADDR", defaultVisionAddr))
	return app, nil
}

func (a *App) EnsureDirs() error {
	for _, dir := range []string{a.DataDir, a.UploadDir, a.PreviewDir} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}
	return nil
}

func (a *App) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", a.handleIndex)
	mux.HandleFunc("/app.js", a.handleStatic)
	mux.HandleFunc("/styles.css", a.handleStatic)
	mux.HandleFunc("/api/health", a.handleHealth)
	mux.HandleFunc("/api/upload", a.handleUpload)
	mux.HandleFunc("/api/analyze", a.handleAnalyze)
	mux.HandleFunc("/api/match", a.handleMatch)
	mux.HandleFunc("/api/match/start", a.handleMatchStart)
	mux.HandleFunc("/api/match/events/", a.handleMatchEvents)
	mux.Handle("/uploads/", http.StripPrefix("/uploads/", http.FileServer(http.Dir(a.UploadDir))))
	mux.Handle("/previews/", http.StripPrefix("/previews/", http.FileServer(http.Dir(a.PreviewDir))))
	return mux
}
