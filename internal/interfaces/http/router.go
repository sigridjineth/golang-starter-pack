package http

import "net/http"

// NewRouter creates a new HTTP router and registers the user routes.
func NewRouter(userHandler *UserHandler) *http.ServeMux {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /users/{id}", userHandler.GetUser)
	mux.HandleFunc("POST /users", userHandler.CreateUser)

	// Add other routes here

	return mux
}
