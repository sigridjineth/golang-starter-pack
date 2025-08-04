package http

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"time"

	"go-starter-template/internal/application"
	"go-starter-template/internal/domain"

	"github.com/go-playground/validator/v10"
)

// Initialize validator
var validate = validator.New()

// CreateUserRequest defines the request body for creating a user.
type CreateUserRequest struct {
	Name  string `json:"name" validate:"required,min=2,max=100"`
	Email string `json:"email" validate:"required,email"`
}

// UserHandler handles HTTP requests for users.
type UserHandler struct {
	service application.UserServiceInterface
}

// NewUserHandler creates a new UserHandler.
func NewUserHandler(service application.UserServiceInterface) *UserHandler {
	return &UserHandler{service: service}
}

// GetUser handles the GET /users/{id} request.
func (h *UserHandler) GetUser(w http.ResponseWriter, r *http.Request) {
	idStr := r.PathValue("id") // Available from Go 1.22
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		http.Error(w, "Invalid user ID", http.StatusBadRequest)
		return
	}

	user, err := h.service.GetUserByID(r.Context(), id)
	if err != nil {
		if errors.Is(err, domain.ErrNotFound) {
			http.Error(w, "User not found", http.StatusNotFound)
			return
		}
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(user); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}

// CreateUser handles the POST /users request.
func (h *UserHandler) CreateUser(w http.ResponseWriter, r *http.Request) {
	var req CreateUserRequest

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate the request body
	if err := validate.Struct(req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// In a real application, you would create the user via the service layer
	// For now, just return the created user
	newUser := &domain.User{
		ID:    time.Now().UnixNano(), // Dummy ID
		Name:  req.Name,
		Email: req.Email,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	if err := json.NewEncoder(w).Encode(newUser); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}
