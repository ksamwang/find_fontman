# Find Fontman SDK

The SDKs are thin clients for the local Fontman runtime service. Callers do not need Python, PyTorch, PaddleOCR, or the font index in their own process.

## Runtime

Build the Windows runtime on the development machine:

```powershell
.\scripts\build_runtime_windows.ps1 -IncludeData -IncludeFonts
```

The output is:

```text
dist/fontman-runtime/
  fontman-service.exe
  data/font_ai/
  fonts/
```

Run it on a Windows machine without Python installed:

```powershell
.\fontman-service.exe --addr 127.0.0.1:9091 --root . --fonts .\fonts --data .\data --previews .\data\previews
```

## Go

Copy `sdk/go/fontman.go` into your Go project.

```go
package main

import (
	"context"
	"fmt"

	"yourapp/fontman"
)

func main() {
	client := fontman.NewClient("http://127.0.0.1:9091")
	res, err := client.Match(context.Background(), fontman.MatchRequest{
		ImagePath: `D:\test\poster.png`,
		Box:       fontman.Box{X: 100, Y: 80, W: 420, H: 120},
		Text:      "品牌标题",
		TopK:      5,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(res.Results[0].FontPath)
}
```

## C++

Copy `sdk/cpp/fontman.hpp` into your C++ project. It is header-only and uses WinHTTP, so link `winhttp.lib`.

```cpp
#include "fontman.hpp"
#include <iostream>

int main() {
    fontman::Client client("http://127.0.0.1:9091");

    fontman::MatchRequest req;
    req.image_path = R"(D:\test\poster.png)";
    req.box = {100, 80, 420, 120};
    req.text = u8"品牌标题";
    req.top_k = 5;

    auto res = client.match(req);
    std::cout << res.results.at(0).font_path << "\n";
}
```

MSVC example:

```powershell
cl /std:c++17 /EHsc main.cpp winhttp.lib
```
