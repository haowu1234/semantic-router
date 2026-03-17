package proxy

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"
)

// NewReverseProxy creates a reverse proxy to targetBase and strips the given prefix from the incoming path
// It also handles CORS, iframe embedding, and other security headers
func NewReverseProxy(targetBase, stripPrefix string, forwardAuth bool) (*httputil.ReverseProxy, error) {
	targetURL, err := url.Parse(targetBase)
	if err != nil {
		return nil, fmt.Errorf("invalid target URL %q: %w", targetBase, err)
	}

	proxy := httputil.NewSingleHostReverseProxy(targetURL)
	proxy.Transport = newProxyTransport()
	proxy.FlushInterval = -1 // flush immediately for SSE streaming

	overrideOrigin := strings.EqualFold(os.Getenv("PROXY_OVERRIDE_ORIGIN"), "true")
	origDirector := proxy.Director
	proxy.Director = makeProxyDirector(origDirector, targetURL, stripPrefix, forwardAuth, overrideOrigin)
	proxy.ErrorHandler = proxyErrorHandler
	proxy.ModifyResponse = proxyModifyResponse

	return proxy, nil
}

func newProxyTransport() *http.Transport {
	return &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   20,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 0, // no limit — SSE streams can be long-lived
		WriteBufferSize:       32 * 1024,
		ReadBufferSize:        32 * 1024,
	}
}

func makeProxyDirector(origDirector func(*http.Request), targetURL *url.URL, stripPrefix string, forwardAuth, overrideOrigin bool) func(*http.Request) {
	return func(r *http.Request) {
		origDirector(r)
		p := rewritePath(r.URL.Path, stripPrefix)
		r.URL.Path = p

		incomingOrigin := extractOrigin(r)
		if incomingOrigin != "" {
			r.Header.Set("X-Forwarded-Origin", incomingOrigin)
		}

		setOriginHeader(r, targetURL, overrideOrigin)
		setForwardedHeaders(r)
		r.Host = targetURL.Host

		if !forwardAuth {
			r.Header.Del("Authorization")
		}
		log.Printf("Proxying: %s %s -> %s://%s%s", r.Method, stripPrefix, targetURL.Scheme, targetURL.Host, p)
	}
}

func rewritePath(path, stripPrefix string) string {
	p := strings.TrimPrefix(path, stripPrefix)
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	return p
}

func extractOrigin(r *http.Request) string {
	origin := r.Header.Get("Origin")
	if origin != "" {
		return origin
	}
	if referer := r.Header.Get("Referer"); referer != "" {
		if refererURL, err := url.Parse(referer); err == nil {
			return refererURL.Scheme + "://" + refererURL.Host
		}
	}
	return ""
}

func setOriginHeader(r *http.Request, targetURL *url.URL, overrideOrigin bool) {
	targetOrigin := targetURL.Scheme + "://" + targetURL.Host
	if overrideOrigin && isWriteMethod(r.Method) {
		r.Header.Set("Origin", targetOrigin)
	} else if r.Header.Get("Origin") == "" {
		r.Header.Set("Origin", targetOrigin)
	}
}

func isWriteMethod(method string) bool {
	return method == http.MethodPost || method == http.MethodPut ||
		method == http.MethodPatch || method == http.MethodDelete
}

func setForwardedHeaders(r *http.Request) {
	r.Header.Set("X-Forwarded-Host", r.Host)

	proto := "http"
	if r.TLS != nil {
		proto = "https"
	}
	if forwardedProto := r.Header.Get("X-Forwarded-Proto"); forwardedProto != "" {
		proto = forwardedProto
	}
	r.Header.Set("X-Forwarded-Proto", proto)

	clientIP := extractClientIP(r.RemoteAddr)
	if clientIP != "" {
		if existing := r.Header.Get("X-Forwarded-For"); existing != "" {
			r.Header.Set("X-Forwarded-For", existing+", "+clientIP)
		} else {
			r.Header.Set("X-Forwarded-For", clientIP)
		}
	}
}

func extractClientIP(remoteAddr string) string {
	if remoteAddr == "" {
		return ""
	}
	ip, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		return remoteAddr
	}
	return ip
}

func proxyErrorHandler(w http.ResponseWriter, r *http.Request, err error) {
	log.Printf("Proxy error for %s: %v", r.URL.Path, err)
	http.Error(w, fmt.Sprintf("Bad Gateway: %v", err), http.StatusBadGateway)
}

func proxyModifyResponse(resp *http.Response) error {
	resp.Header.Del("X-Frame-Options")
	updateCSPHeader(resp)
	setCORSHeaders(resp)
	resp.Header.Set("Access-Control-Allow-Private-Network", "true")
	return nil
}

func updateCSPHeader(resp *http.Response) {
	csp := resp.Header.Get("Content-Security-Policy")
	if csp == "" {
		resp.Header.Set("Content-Security-Policy", "frame-ancestors 'self'")
		return
	}
	if strings.Contains(strings.ToLower(csp), "frame-ancestors") {
		parts := strings.Split(csp, ";")
		for i, d := range parts {
			if strings.Contains(strings.ToLower(d), "frame-ancestors") {
				parts[i] = "frame-ancestors 'self'"
			}
		}
		resp.Header.Set("Content-Security-Policy", strings.Join(parts, ";"))
	} else {
		resp.Header.Set("Content-Security-Policy", csp+"; frame-ancestors 'self'")
	}
}

func setCORSHeaders(resp *http.Response) {
	origin := resp.Request.Header.Get("X-Forwarded-Origin")
	if origin != "" {
		resp.Header.Set("Access-Control-Allow-Origin", origin)
		resp.Header.Set("Access-Control-Allow-Credentials", "true")
		resp.Header.Set("Vary", "Origin")
	} else {
		resp.Header.Set("Access-Control-Allow-Origin", "*")
	}
	resp.Header.Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
	resp.Header.Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, Accept, Origin")
	resp.Header.Set("Access-Control-Expose-Headers", "Content-Length, Content-Range")
}

// NewJaegerProxy creates a reverse proxy specifically for Jaeger UI with dark theme injection
func NewJaegerProxy(targetBase, stripPrefix string) (*httputil.ReverseProxy, error) {
	proxy, err := NewReverseProxy(targetBase, stripPrefix, false)
	if err != nil {
		return nil, err
	}

	// Override ModifyResponse to inject dark theme script into HTML responses
	originalModifyResponse := proxy.ModifyResponse
	proxy.ModifyResponse = func(resp *http.Response) error {
		// First apply the original response modifications (CORS, CSP, etc.)
		if originalModifyResponse != nil {
			if err := originalModifyResponse(resp); err != nil {
				return err
			}
		}

		// Only inject script into HTML responses
		contentType := resp.Header.Get("Content-Type")
		if !strings.Contains(contentType, "text/html") {
			return nil
		}

		// Read the response body
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		resp.Body.Close()

		// Inject light theme script to ensure Jaeger displays consistently in light mode
		// This avoids theme conflicts with the dashboard
		themeScript := `<script>
(function() {
  try {
    // Force Jaeger UI to use light theme for consistent appearance
    localStorage.setItem('jaeger-ui-theme', 'light');
    localStorage.setItem('theme', 'light');

    // Set data-theme attribute on document element
    if (document.documentElement) {
      document.documentElement.setAttribute('data-theme', 'light');
      document.documentElement.setAttribute('data-bs-theme', 'light');
      document.documentElement.style.colorScheme = 'light';
    }

    // Also set it after DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
      if (document.documentElement) {
        document.documentElement.setAttribute('data-theme', 'light');
        document.documentElement.setAttribute('data-bs-theme', 'light');
        document.documentElement.style.colorScheme = 'light';
      }
    });
  } catch (e) {
    console.error('Failed to set Jaeger theme:', e);
  }
})();
</script>`

		// Try to inject before </head>, otherwise before </body>
		modifiedBody := string(body)
		if strings.Contains(modifiedBody, "</head>") {
			modifiedBody = strings.Replace(modifiedBody, "</head>", themeScript+"</head>", 1)
		} else if strings.Contains(modifiedBody, "<body") {
			// Find the end of the <body> tag and inject after it
			bodyTagEnd := strings.Index(modifiedBody, ">")
			if bodyTagEnd != -1 {
				modifiedBody = modifiedBody[:bodyTagEnd+1] + themeScript + modifiedBody[bodyTagEnd+1:]
			}
		}

		// Create new response body
		newBody := []byte(modifiedBody)
		resp.Body = io.NopCloser(bytes.NewReader(newBody))
		resp.ContentLength = int64(len(newBody))
		resp.Header.Set("Content-Length", fmt.Sprintf("%d", len(newBody)))

		return nil
	}

	return proxy, nil
}
