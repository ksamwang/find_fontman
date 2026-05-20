package fontman

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const DefaultBaseURL = "http://127.0.0.1:9091"

type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

type Box struct {
	X int `json:"x"`
	Y int `json:"y"`
	W int `json:"w"`
	H int `json:"h"`
}

type HealthResponse struct {
	OK           bool           `json:"ok"`
	Pillow       bool           `json:"pillow"`
	NumPy        bool           `json:"numpy"`
	PaddleOCR    bool           `json:"paddleocr"`
	FontCount    int            `json:"font_count"`
	MatchMode    string         `json:"match_mode"`
	Capabilities map[string]any `json:"capabilities"`
}

type AnalyzeRequest struct {
	ImagePath string `json:"image_path"`
	Box       Box    `json:"box"`
}

type AnalyzeResponse struct {
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
	CropURL    string  `json:"crop_url"`
	Warning    string  `json:"warning,omitempty"`
}

type MatchRequest struct {
	ImagePath string `json:"image_path"`
	Box       Box    `json:"box"`
	Text      string `json:"text"`
	TopK      int    `json:"top_k,omitempty"`
	Rerank    bool   `json:"rerank,omitempty"`
}

type MatchResponse struct {
	Results       []FontResult `json:"results"`
	CandidateSize int          `json:"candidate_size"`
	ElapsedMS     int64        `json:"elapsed_ms"`
	Warning       string       `json:"warning,omitempty"`
	MatchMode     string       `json:"match_mode,omitempty"`
	TopIndices    []int        `json:"top_indices,omitempty"`
}

type FontResult struct {
	FontName       string         `json:"font_name"`
	FontPath       string         `json:"font_path"`
	ScoreTotal     float64        `json:"score_total"`
	ScoreSSIM      float64        `json:"score_ssim"`
	ScoreIOU       float64        `json:"score_iou"`
	ScoreEdge      float64        `json:"score_edge"`
	ScoreShape     float64        `json:"score_shape"`
	ScoreChamfer   float64        `json:"score_chamfer"`
	ScoreDensity   float64        `json:"score_density"`
	EmbeddingScore float64        `json:"embedding_score"`
	MatchMode      string         `json:"match_mode"`
	Align          map[string]any `json:"align"`
	PreviewPath    string         `json:"preview_path"`
	PreviewBase64  string         `json:"preview_base64"`
	PreviewMIME    string         `json:"preview_mime"`
}

func NewClient(baseURL string) *Client {
	if strings.TrimSpace(baseURL) == "" {
		baseURL = DefaultBaseURL
	}
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		HTTPClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

func (c *Client) Health(ctx context.Context) (*HealthResponse, error) {
	var out HealthResponse
	if err := c.getJSON(ctx, "/health", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) Analyze(ctx context.Context, req AnalyzeRequest) (*AnalyzeResponse, error) {
	var out AnalyzeResponse
	if err := c.postJSON(ctx, "/analyze", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) Match(ctx context.Context, req MatchRequest) (*MatchResponse, error) {
	req.Text = strings.TrimSpace(req.Text)
	if req.TopK <= 0 {
		req.TopK = 10
	}
	var out MatchResponse
	if err := c.postJSON(ctx, "/match", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) getJSON(ctx context.Context, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+path, nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return decodeResponse(resp, out)
}

func (c *Client) postJSON(ctx context.Context, path string, payload any, out any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.httpClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return decodeResponse(resp, out)
}

func (c *Client) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return http.DefaultClient
}

func decodeResponse(resp *http.Response, out any) error {
	if resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		return fmt.Errorf("fontman status %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}
