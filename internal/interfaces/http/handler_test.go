package http_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"go-starter-template/internal/domain"
	interfaces "go-starter-template/internal/interfaces/http"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockUserService is a mock implementation of application.UserServiceInterface
type MockUserService struct {
	mock.Mock
}

// GetUserByID mocks the GetUserByID method of UserServiceInterface
func (m *MockUserService) GetUserByID(ctx context.Context, id int64) (*domain.User, error) {
	args := m.Called(ctx, id)
	return args.Get(0).(*domain.User), args.Error(1)
}

func TestUserHandler_CreateUser(t *testing.T) {
	tests := []struct {
		name           string
		requestBody    map[string]interface{}
		expectedStatus int
		expectedBody   string
	}{
		{
			name:           "valid request",
			requestBody:    map[string]interface{}{"name": "John Doe", "email": "john.doe@example.com"},
			expectedStatus: http.StatusCreated,
			expectedBody:   "{\"id\":", // Check for partial match as ID is dynamic
		},
		{
			name:           "invalid email",
			requestBody:    map[string]interface{}{"name": "John Doe", "email": "invalid-email"},
			expectedStatus: http.StatusBadRequest,
			expectedBody:   "Key: 'CreateUserRequest.Email' Error:Field validation for 'Email' failed on the 'email' tag",
		},
		{
			name:           "missing name",
			requestBody:    map[string]interface{}{"email": "john.doe@example.com"},
			expectedStatus: http.StatusBadRequest,
			expectedBody:   "Key: 'CreateUserRequest.Name' Error:Field validation for 'Name' failed on the 'required' tag",
		},
		{
			name:           "empty request body",
			requestBody:    map[string]interface{}{},
			expectedStatus: http.StatusBadRequest,
			expectedBody:   "Key: 'CreateUserRequest.Name' Error:Field validation for 'Name' failed on the 'required' tag",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a mock UserService (not used in this specific test, but good practice)
			mockService := new(MockUserService)
			handler := interfaces.NewUserHandler(mockService)

			// Prepare the request body
			requestBodyBytes, _ := json.Marshal(tt.requestBody)
			req := httptest.NewRequest(http.MethodPost, "/users", bytes.NewBuffer(requestBodyBytes))
			req.Header.Set("Content-Type", "application/json")

			// Create a ResponseRecorder to record the response
			rr := httptest.NewRecorder()

			// Call the handler
			handler.CreateUser(rr, req)

			// Assert the status code
			assert.Equal(t, tt.expectedStatus, rr.Code)

			// Assert the response body (partial match for dynamic ID)
			assert.Contains(t, rr.Body.String(), tt.expectedBody)
		})
	}
}
