package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type VisionClient struct {
	app     *App
	BaseURL string
	cmd     *exec.Cmd
}

func NewVisionClient(app *App, baseURL string) *VisionClient {
	return &VisionClient{app: app, BaseURL: baseURL}
}

func (a *App) StartVisionService() error {
	a.visionMu.Lock()
	defer a.visionMu.Unlock()

	if a.Vision.Check(context.Background()) == nil {
		return nil
	}

	python := env("PYTHON", filepath.Join(a.Root, ".venv", "Scripts", "python.exe"))
	if _, err := os.Stat(python); err != nil {
		python = env("PYTHON_FALLBACK", "python")
	}
	script := filepath.Join(a.Root, "python_service", "service.py")
	cmd := exec.Command(python, script,
		"--addr", env("VISION_ADDR", defaultVisionAddr),
		"--root", a.Root,
		"--fonts", a.FontsDir,
		"--data", a.DataDir,
		"--previews", a.PreviewDir,
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Dir = a.Root
	if err := cmd.Start(); err != nil {
		return err
	}
	a.Vision.cmd = cmd
	go func() {
		err := cmd.Wait()
		logVisionExit(err)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), visionStartTimeout())
	defer cancel()
	var lastErr error
	for ctx.Err() == nil {
		if err := a.Vision.Check(ctx); err == nil {
			return nil
		} else {
			lastErr = err
		}
		time.Sleep(300 * time.Millisecond)
	}
	if lastErr != nil {
		return fmt.Errorf("python vision service did not become healthy: %w", lastErr)
	}
	return errors.New("python vision service did not become healthy")
}

func visionStartTimeout() time.Duration {
	raw := env("VISION_START_TIMEOUT", "")
	if raw == "" {
		return 90 * time.Second
	}
	if seconds, err := strconv.Atoi(raw); err == nil {
		return time.Duration(seconds) * time.Second
	}
	if duration, err := time.ParseDuration(raw); err == nil {
		return duration
	}
	return 90 * time.Second
}

func (v *VisionClient) Check(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, v.BaseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("vision health status %d", resp.StatusCode)
	}
	var health struct {
		Capabilities struct {
			PreviewBase64 bool `json:"preview_base64"`
		} `json:"capabilities"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return err
	}
	if !health.Capabilities.PreviewBase64 {
		return errors.New("vision service is missing preview_base64 capability")
	}
	return nil
}

func (v *VisionClient) Call(ctx context.Context, path string, payload any, out any) error {
	if err := v.app.StartVisionService(); err != nil {
		return err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, v.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("vision %s failed: %s", path, strings.TrimSpace(string(msg)))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func (v *VisionClient) Stream(ctx context.Context, path string) (*http.Response, error) {
	if err := v.app.StartVisionService(); err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, v.BaseURL+path, nil)
	if err != nil {
		return nil, err
	}
	return http.DefaultClient.Do(req)
}

func logVisionExit(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "python vision service exited: %v\n", err)
	}
}
