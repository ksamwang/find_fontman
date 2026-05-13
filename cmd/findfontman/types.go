package main

type uploadResponse struct {
	ID  string `json:"id"`
	URL string `json:"url"`
}

type box struct {
	X int `json:"x"`
	Y int `json:"y"`
	W int `json:"w"`
	H int `json:"h"`
}

type analyzeRequest struct {
	ImageID string `json:"image_id"`
	Box     box    `json:"box"`
}

type analyzeResponse struct {
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
	CropURL    string  `json:"crop_url"`
	Warning    string  `json:"warning,omitempty"`
}

type matchRequest struct {
	ImageID string `json:"image_id"`
	Box     box    `json:"box"`
	Text    string `json:"text"`
	TopK    int    `json:"top_k"`
}

type scoreBreakdown struct {
	SSIM  float64 `json:"ssim"`
	IOU   float64 `json:"iou"`
	Edge  float64 `json:"edge"`
	Shape float64 `json:"shape"`
}

type fontResult struct {
	FontName     string         `json:"font_name"`
	FontPath     string         `json:"font_path"`
	ScoreTotal   float64        `json:"score_total"`
	ScoreSSIM    float64        `json:"score_ssim"`
	ScoreIOU     float64        `json:"score_iou"`
	ScoreEdge    float64        `json:"score_edge"`
	ScoreShape   float64        `json:"score_shape"`
	PreviewPath  string         `json:"preview_path"`
	PreviewURL   string         `json:"preview_url"`
	ScoreDetails scoreBreakdown `json:"score_details"`
}

type matchResponse struct {
	Results       []fontResult `json:"results"`
	CandidateSize int          `json:"candidate_size"`
	ElapsedMS     int64        `json:"elapsed_ms"`
	Warning       string       `json:"warning,omitempty"`
}

type matchStartResponse struct {
	TaskID string `json:"task_id"`
}
