#pragma once

#ifndef _WIN32
#error "fontman.hpp currently supports Windows only."
#endif

#include <windows.h>
#include <winhttp.h>

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#pragma comment(lib, "winhttp.lib")

namespace fontman {

struct Box {
    int x = 0;
    int y = 0;
    int w = 1;
    int h = 1;
};

struct FontResult {
    std::string font_name;
    std::string font_path;
    double score_total = 0.0;
    double embedding_score = 0.0;
    std::string match_mode;
    std::string preview_base64;
    std::string preview_mime;
};

struct MatchRequest {
    std::string image_path;
    Box box;
    std::string text;
    int top_k = 10;
    bool rerank = false;
};

struct MatchResponse {
    std::vector<FontResult> results;
    int candidate_size = 0;
    long long elapsed_ms = 0;
    std::string warning;
    std::string match_mode;
    std::string raw_json;
};

struct HealthResponse {
    bool ok = false;
    int font_count = 0;
    std::string match_mode;
    std::string raw_json;
};

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& message) : std::runtime_error(message) {}
};

class Client {
public:
    explicit Client(const std::string& base_url = "http://127.0.0.1:9091")
        : base_url_(trim_right_slash(base_url)) {}

    HealthResponse health() const {
        std::string body = request(L"GET", L"/health", "");
        HealthResponse out;
        out.raw_json = body;
        out.ok = json_bool(body, "ok");
        out.font_count = static_cast<int>(json_number(body, "font_count"));
        out.match_mode = json_string(body, "match_mode");
        return out;
    }

    MatchResponse match(const MatchRequest& req) const {
        std::string body = request(L"POST", L"/match", match_payload(req));
        MatchResponse out;
        out.raw_json = body;
        out.candidate_size = static_cast<int>(json_number(body, "candidate_size"));
        out.elapsed_ms = static_cast<long long>(json_number(body, "elapsed_ms"));
        out.warning = json_string(body, "warning");
        out.match_mode = json_string(body, "match_mode");
        parse_results(body, out.results);
        return out;
    }

private:
    std::string base_url_;

    struct ParsedURL {
        std::wstring host;
        INTERNET_PORT port = 0;
        bool https = false;
    };

    std::string request(const wchar_t* method, const wchar_t* path, const std::string& payload) const {
        ParsedURL url = parse_base_url(base_url_);
        HINTERNET session = WinHttpOpen(L"fontman-cpp/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        if (!session) throw Error("WinHttpOpen failed");
        Handle session_handle(session);

        HINTERNET connect = WinHttpConnect(session, url.host.c_str(), url.port, 0);
        if (!connect) throw Error("WinHttpConnect failed");
        Handle connect_handle(connect);

        DWORD flags = url.https ? WINHTTP_FLAG_SECURE : 0;
        HINTERNET request_handle_raw = WinHttpOpenRequest(connect, method, path, nullptr,
                                                          WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
        if (!request_handle_raw) throw Error("WinHttpOpenRequest failed");
        Handle request_handle(request_handle_raw);

        std::wstring headers = L"Content-Type: application/json\r\nAccept: application/json\r\n";
        BOOL ok = WinHttpSendRequest(
            request_handle_raw,
            headers.c_str(),
            static_cast<DWORD>(headers.size()),
            payload.empty() ? WINHTTP_NO_REQUEST_DATA : const_cast<char*>(payload.data()),
            static_cast<DWORD>(payload.size()),
            static_cast<DWORD>(payload.size()),
            0);
        if (!ok) throw Error("WinHttpSendRequest failed");
        if (!WinHttpReceiveResponse(request_handle_raw, nullptr)) throw Error("WinHttpReceiveResponse failed");

        DWORD status = 0;
        DWORD status_size = sizeof(status);
        WinHttpQueryHeaders(request_handle_raw, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                            WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size, WINHTTP_NO_HEADER_INDEX);

        std::string response;
        while (true) {
            DWORD available = 0;
            if (!WinHttpQueryDataAvailable(request_handle_raw, &available)) throw Error("WinHttpQueryDataAvailable failed");
            if (available == 0) break;
            std::string chunk(available, '\0');
            DWORD read = 0;
            if (!WinHttpReadData(request_handle_raw, chunk.data(), available, &read)) throw Error("WinHttpReadData failed");
            chunk.resize(read);
            response += chunk;
        }
        if (status >= 300) {
            throw Error("fontman status " + std::to_string(status) + ": " + response);
        }
        return response;
    }

    class Handle {
    public:
        explicit Handle(HINTERNET value) : value_(value) {}
        ~Handle() {
            if (value_) WinHttpCloseHandle(value_);
        }
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;

    private:
        HINTERNET value_;
    };

    static ParsedURL parse_base_url(const std::string& raw) {
        std::wstring wide = utf8_to_wide(raw);
        URL_COMPONENTS parts{};
        parts.dwStructSize = sizeof(parts);
        wchar_t host[256]{};
        parts.lpszHostName = host;
        parts.dwHostNameLength = static_cast<DWORD>(sizeof(host) / sizeof(host[0]));
        if (!WinHttpCrackUrl(wide.c_str(), static_cast<DWORD>(wide.size()), 0, &parts)) {
            throw Error("invalid base URL: " + raw);
        }
        ParsedURL out;
        out.host.assign(parts.lpszHostName, parts.dwHostNameLength);
        out.port = parts.nPort;
        out.https = parts.nScheme == INTERNET_SCHEME_HTTPS;
        return out;
    }

    static std::string match_payload(const MatchRequest& req) {
        std::string json = "{";
        json += "\"image_path\":\"" + escape_json(req.image_path) + "\",";
        json += "\"box\":{\"x\":" + std::to_string(req.box.x) + ",\"y\":" + std::to_string(req.box.y) +
                ",\"w\":" + std::to_string(req.box.w) + ",\"h\":" + std::to_string(req.box.h) + "},";
        json += "\"text\":\"" + escape_json(req.text) + "\",";
        json += "\"top_k\":" + std::to_string(req.top_k) + ",";
        json += "\"rerank\":" + std::string(req.rerank ? "true" : "false");
        json += "}";
        return json;
    }

    static void parse_results(const std::string& json, std::vector<FontResult>& results) {
        std::size_t key = json.find("\"results\"");
        if (key == std::string::npos) return;
        std::size_t array = json.find('[', key);
        if (array == std::string::npos) return;
        std::size_t pos = array + 1;
        while (true) {
            std::size_t obj_start = json.find('{', pos);
            if (obj_start == std::string::npos) break;
            std::size_t obj_end = matching_brace(json, obj_start);
            if (obj_end == std::string::npos) break;
            std::string obj = json.substr(obj_start, obj_end - obj_start + 1);
            FontResult item;
            item.font_name = json_string(obj, "font_name");
            item.font_path = json_string(obj, "font_path");
            item.score_total = json_number(obj, "score_total");
            item.embedding_score = json_number(obj, "embedding_score");
            item.match_mode = json_string(obj, "match_mode");
            item.preview_base64 = json_string(obj, "preview_base64");
            item.preview_mime = json_string(obj, "preview_mime");
            results.push_back(item);
            pos = obj_end + 1;
        }
    }

    static std::size_t matching_brace(const std::string& text, std::size_t start) {
        int depth = 0;
        bool in_string = false;
        bool escaped = false;
        for (std::size_t i = start; i < text.size(); ++i) {
            char ch = text[i];
            if (in_string) {
                if (escaped) {
                    escaped = false;
                } else if (ch == '\\') {
                    escaped = true;
                } else if (ch == '"') {
                    in_string = false;
                }
                continue;
            }
            if (ch == '"') in_string = true;
            if (ch == '{') ++depth;
            if (ch == '}' && --depth == 0) return i;
        }
        return std::string::npos;
    }

    static std::string json_string(const std::string& json, const std::string& name) {
        std::size_t pos = value_pos(json, name);
        if (pos == std::string::npos || json[pos] != '"') return "";
        ++pos;
        std::string out;
        bool escaped = false;
        for (; pos < json.size(); ++pos) {
            char ch = json[pos];
            if (escaped) {
                switch (ch) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                default: out += ch; break;
                }
                escaped = false;
                continue;
            }
            if (ch == '\\') {
                escaped = true;
                continue;
            }
            if (ch == '"') break;
            out += ch;
        }
        return out;
    }

    static double json_number(const std::string& json, const std::string& name) {
        std::size_t pos = value_pos(json, name);
        if (pos == std::string::npos) return 0.0;
        char* end = nullptr;
        return std::strtod(json.c_str() + pos, &end);
    }

    static bool json_bool(const std::string& json, const std::string& name) {
        std::size_t pos = value_pos(json, name);
        if (pos == std::string::npos) return false;
        return json.compare(pos, 4, "true") == 0;
    }

    static std::size_t value_pos(const std::string& json, const std::string& name) {
        std::string key = "\"" + name + "\"";
        std::size_t pos = json.find(key);
        if (pos == std::string::npos) return std::string::npos;
        pos = json.find(':', pos + key.size());
        if (pos == std::string::npos) return std::string::npos;
        ++pos;
        while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
        return pos;
    }

    static std::string escape_json(const std::string& value) {
        std::string out;
        for (char ch : value) {
            switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += ch; break;
            }
        }
        return out;
    }

    static std::wstring utf8_to_wide(const std::string& value) {
        if (value.empty()) return L"";
        int size = MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0);
        std::wstring out(size, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), out.data(), size);
        return out;
    }

    static std::string trim_right_slash(std::string value) {
        while (!value.empty() && value.back() == '/') value.pop_back();
        return value;
    }
};

}  // namespace fontman
