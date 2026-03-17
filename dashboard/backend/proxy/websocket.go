package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// NewWebSocketAwareHandler returns an http.Handler that proxies both regular HTTP
// and WebSocket upgrade requests to the target. This is required for services
// like OpenClaw whose control UI uses WebSocket for real-time communication.
func NewWebSocketAwareHandler(targetBase, stripPrefix string) (http.Handler, error) {
	return NewWebSocketAwareHandlerWithHeaders(targetBase, stripPrefix, nil)
}

// NewWebSocketAwareHandlerWithHeaders behaves like NewWebSocketAwareHandler but
// also injects static headers into all proxied HTTP and WebSocket upgrade requests.
func NewWebSocketAwareHandlerWithHeaders(targetBase, stripPrefix string, staticHeaders map[string]string) (http.Handler, error) {
	targetURL, err := url.Parse(targetBase)
	if err != nil {
		return nil, err
	}

	httpProxy, err := NewReverseProxy(targetBase, stripPrefix, false)
	if err != nil {
		return nil, err
	}
	if len(staticHeaders) > 0 {
		origDirector := httpProxy.Director
		httpProxy.Director = func(r *http.Request) {
			origDirector(r)
			for key, value := range staticHeaders {
				normalizedKey := strings.TrimSpace(key)
				normalizedValue := strings.TrimSpace(value)
				if normalizedKey == "" || normalizedValue == "" {
					continue
				}
				// Static headers are authoritative in embedded mode to avoid stale
				// client-side gateway tokens causing auth mismatch loops.
				r.Header.Set(normalizedKey, normalizedValue)
			}
		}
	}
	origModify := httpProxy.ModifyResponse
	httpProxy.ModifyResponse = func(resp *http.Response) error {
		if origModify != nil {
			if err := origModify(resp); err != nil {
				return err
			}
		}

		// Rewrite control-ui-config.json to set basePath for embedded mode.
		if strings.HasSuffix(resp.Request.URL.Path, "/__openclaw/control-ui-config.json") {
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				return err
			}
			_ = resp.Body.Close()

			var cfg map[string]interface{}
			if err := json.Unmarshal(body, &cfg); err == nil {
				embeddedBasePath := strings.TrimRight(strings.TrimSpace(stripPrefix), "/")
				if embeddedBasePath == "" {
					embeddedBasePath = "/"
				}

				// Always force embedded basePath. OpenClaw may default basePath to "/",
				// which breaks WebSocket routing when loaded behind /embedded/openclaw/{name}/.
				cfg["basePath"] = embeddedBasePath

				// Provide a relative gateway URL so Control UI resolves WebSocket requests
				// through the embedded proxy path (no host/port exposure in client config).
				cfg["gatewayUrl"] = embeddedBasePath
				updated, err := json.Marshal(cfg)
				if err == nil {
					resp.Body = io.NopCloser(bytes.NewReader(updated))
					resp.ContentLength = int64(len(updated))
					resp.Header.Set("Content-Length", strconv.Itoa(len(updated)))
					resp.Header.Set("Content-Type", "application/json; charset=utf-8")
				} else {
					resp.Body = io.NopCloser(bytes.NewReader(body))
				}
			} else {
				resp.Body = io.NopCloser(bytes.NewReader(body))
			}
			return nil
		}
		return nil
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if isWebSocketUpgrade(r) {
			proxyWebSocket(w, r, targetURL, stripPrefix, staticHeaders)
			return
		}
		httpProxy.ServeHTTP(w, r)
	}), nil
}

func isWebSocketUpgrade(r *http.Request) bool {
	for _, v := range r.Header.Values("Connection") {
		for _, token := range strings.Split(v, ",") {
			if strings.EqualFold(strings.TrimSpace(token), "upgrade") {
				if strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
					return true
				}
			}
		}
	}
	return false
}

func proxyWebSocket(w http.ResponseWriter, r *http.Request, target *url.URL, stripPrefix string, staticHeaders map[string]string) {
	targetHost := resolveTargetHost(target)
	path := rewritePath(r.URL.Path, stripPrefix)

	targetConn, err := net.DialTimeout("tcp", targetHost, 10*time.Second)
	if err != nil {
		log.Printf("WebSocket proxy: failed to connect to %s: %v", targetHost, err)
		http.Error(w, "Bad Gateway", http.StatusBadGateway)
		return
	}
	defer targetConn.Close()

	const wsIdleTimeout = 30 * time.Minute
	_ = targetConn.SetDeadline(time.Now().Add(wsIdleTimeout))

	clientConn, clientBuf, err := hijackClientConnection(w)
	if err != nil {
		log.Printf("WebSocket proxy: %v", err)
		http.Error(w, "WebSocket proxy error", http.StatusInternalServerError)
		return
	}
	defer clientConn.Close()
	_ = clientConn.SetDeadline(time.Now().Add(wsIdleTimeout))

	upgradeReq := buildUpgradeRequest(r, target, path, staticHeaders)
	if _, err := targetConn.Write([]byte(upgradeReq)); err != nil {
		log.Printf("WebSocket proxy: failed to write upgrade request: %v", err)
		return
	}

	log.Printf("WebSocket proxy: %s %s -> %s%s", r.Method, r.URL.Path, target.Host, path)
	bidirectionalCopy(targetConn, clientConn, clientBuf, wsIdleTimeout)
}

func resolveTargetHost(target *url.URL) string {
	targetHost := target.Host
	if strings.Contains(targetHost, ":") {
		return targetHost
	}
	if target.Scheme == "https" || target.Scheme == "wss" {
		return targetHost + ":443"
	}
	return targetHost + ":80"
}

func hijackClientConnection(w http.ResponseWriter) (net.Conn, *bufio.ReadWriter, error) {
	hijacker, ok := w.(http.Hijacker)
	if !ok {
		return nil, nil, fmt.Errorf("hijacking not supported")
	}
	clientConn, clientBuf, err := hijacker.Hijack()
	if err != nil {
		return nil, nil, fmt.Errorf("hijack failed: %v", err)
	}
	return clientConn, clientBuf, nil
}

func buildUpgradeRequest(r *http.Request, target *url.URL, path string, staticHeaders map[string]string) string {
	reqURL := path
	if r.URL.RawQuery != "" {
		reqURL += "?" + r.URL.RawQuery
	}

	var reqBuf strings.Builder
	reqBuf.WriteString(r.Method + " " + reqURL + " HTTP/1.1\r\n")
	reqBuf.WriteString("Host: " + target.Host + "\r\n")

	effectiveHeaders := normalizeStaticHeaders(staticHeaders)
	overriddenKeys := makeOverriddenKeySet(effectiveHeaders)
	writeOriginalHeaders(&reqBuf, r.Header, overriddenKeys)
	writeStaticHeaders(&reqBuf, effectiveHeaders)
	reqBuf.WriteString("\r\n")

	return reqBuf.String()
}

func normalizeStaticHeaders(staticHeaders map[string]string) map[string]string {
	effective := make(map[string]string)
	for key, value := range staticHeaders {
		k := strings.TrimSpace(key)
		v := strings.TrimSpace(value)
		if k != "" && v != "" {
			effective[k] = v
		}
	}
	return effective
}

func makeOverriddenKeySet(effectiveHeaders map[string]string) map[string]struct{} {
	overridden := make(map[string]struct{})
	for key := range effectiveHeaders {
		overridden[strings.ToLower(strings.TrimSpace(key))] = struct{}{}
	}
	return overridden
}

func writeOriginalHeaders(buf *strings.Builder, headers http.Header, overridden map[string]struct{}) {
	for key, vals := range headers {
		if strings.EqualFold(key, "Host") {
			continue
		}
		if _, skip := overridden[strings.ToLower(strings.TrimSpace(key))]; skip {
			continue
		}
		for _, val := range vals {
			buf.WriteString(key + ": " + val + "\r\n")
		}
	}
}

func writeStaticHeaders(buf *strings.Builder, headers map[string]string) {
	for key, value := range headers {
		buf.WriteString(key + ": " + value + "\r\n")
	}
}

func bidirectionalCopy(targetConn, clientConn net.Conn, clientBuf *bufio.ReadWriter, idleTimeout time.Duration) {
	done := make(chan struct{}, 2)

	copyFn := func(dst net.Conn, src io.Reader) {
		buf := make([]byte, 32*1024)
		for {
			n, err := src.Read(buf)
			if n > 0 {
				_ = targetConn.SetDeadline(time.Now().Add(idleTimeout))
				_ = clientConn.SetDeadline(time.Now().Add(idleTimeout))
				if _, wErr := dst.Write(buf[:n]); wErr != nil {
					break
				}
			}
			if err != nil {
				break
			}
		}
		done <- struct{}{}
	}

	go copyFn(targetConn, clientBuf)
	go copyFn(clientConn, targetConn)
	<-done
}
