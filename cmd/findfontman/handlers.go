package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func (a *App) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{
		"ok":          true,
		"vision_base": a.Vision.BaseURL,
		"fonts_dir":   a.FontsDir,
	})
}

func (a *App) handleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	file, header, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "missing image", http.StatusBadRequest)
		return
	}
	defer file.Close()

	id := strconv.FormatInt(time.Now().UnixNano(), 36)
	ext := strings.ToLower(filepath.Ext(header.Filename))
	if ext == "" {
		ext = ".png"
	}
	if !allowedImageExt(ext) {
		http.Error(w, "unsupported image type", http.StatusBadRequest)
		return
	}
	dst := filepath.Join(a.UploadDir, id+ext)
	if err := saveMultipart(file, dst); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, uploadResponse{ID: id + ext, URL: "/uploads/" + id + ext})
}

func (a *App) handleAnalyze(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req analyzeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	imagePath, err := a.imagePath(req.ImageID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	payload := map[string]any{"image_path": imagePath, "box": req.Box}
	var out analyzeResponse
	if err := a.Vision.Call(r.Context(), "/analyze", payload, &out); err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, out)
}

func (a *App) handleMatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	payload, ok := a.matchPayload(w, r)
	if !ok {
		return
	}
	var out matchResponse
	if err := a.Vision.Call(r.Context(), "/match", payload, &out); err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	for i := range out.Results {
		if out.Results[i].PreviewPath != "" {
			out.Results[i].PreviewURL = "/previews/" + filepath.Base(out.Results[i].PreviewPath)
		}
		out.Results[i].ScoreDetails = scoreBreakdown{
			SSIM:  out.Results[i].ScoreSSIM,
			IOU:   out.Results[i].ScoreIOU,
			Edge:  out.Results[i].ScoreEdge,
			Shape: out.Results[i].ScoreShape,
		}
	}
	writeJSON(w, out)
}

func (a *App) handleMatchStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	payload, ok := a.matchPayload(w, r)
	if !ok {
		return
	}
	var out matchStartResponse
	if err := a.Vision.Call(r.Context(), "/match/start", payload, &out); err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, out)
}

func (a *App) handleMatchEvents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	taskID := strings.TrimPrefix(r.URL.Path, "/api/match/events/")
	if taskID == "" || strings.Contains(taskID, "/") {
		http.Error(w, "invalid task id", http.StatusBadRequest)
		return
	}
	resp, err := a.Vision.Stream(r.Context(), "/match/events/"+taskID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		http.Error(w, strings.TrimSpace(string(msg)), http.StatusBadGateway)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		_, _ = w.Write(scanner.Bytes())
		_, _ = w.Write([]byte("\n"))
		flusher.Flush()
	}
}

func (a *App) matchPayload(w http.ResponseWriter, r *http.Request) (map[string]any, bool) {
	var req matchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return nil, false
	}
	req.Text = strings.TrimSpace(req.Text)
	if req.Text == "" {
		http.Error(w, "text is required", http.StatusBadRequest)
		return nil, false
	}
	if req.TopK <= 0 || req.TopK > 50 {
		req.TopK = 10
	}
	imagePath, err := a.imagePath(req.ImageID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return nil, false
	}
	return map[string]any{"image_path": imagePath, "box": req.Box, "text": req.Text, "top_k": req.TopK}, true
}

func (a *App) imagePath(id string) (string, error) {
	id = filepath.Base(id)
	if id == "." || id == string(filepath.Separator) || id == "" {
		return "", errors.New("invalid image id")
	}
	path := filepath.Join(a.UploadDir, id)
	clean := filepath.Clean(path)
	if !strings.HasPrefix(strings.ToLower(clean), strings.ToLower(filepath.Clean(a.UploadDir))) {
		return "", errors.New("invalid image path")
	}
	if _, err := os.Stat(clean); err != nil {
		return "", err
	}
	return clean, nil
}

func saveMultipart(file multipart.File, dst string) error {
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, file)
	return err
}

func allowedImageExt(ext string) bool {
	switch ext {
	case ".png", ".jpg", ".jpeg", ".webp", ".bmp":
		return true
	default:
		return false
	}
}
