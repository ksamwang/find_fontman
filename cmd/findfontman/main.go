package main

import (
	"log"
	"net/http"
	"os"
)

func main() {
	app, err := NewApp()
	if err != nil {
		log.Fatal(err)
	}
	if err := app.EnsureDirs(); err != nil {
		log.Fatal(err)
	}
	if env("SKIP_PYTHON_SERVICE", "") != "1" {
		if err := app.StartVisionService(); err != nil {
			log.Printf("python vision service not started: %v", err)
		}
	}

	addr := env("ADDR", defaultAddr)
	log.Printf("find_fontman listening on http://localhost%s", addr)
	log.Fatal(http.ListenAndServe(addr, logRequest(app.Routes())))
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
