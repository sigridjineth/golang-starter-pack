package application_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"go-starter-template/internal/application"
	"go-starter-template/internal/domain"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"go.uber.org/fx"
	"go.uber.org/fx/fxtest"
)

// MockUserRepository is a mock implementation of domain.UserRepository
type MockUserRepository struct {
	mock.Mock
}

// FindByID mocks the FindByID method of UserRepository
func (m *MockUserRepository) FindByID(ctx context.Context, id int64) (*domain.User, error) {
	args := m.Called(ctx, id)
	return args.Get(0).(*domain.User), args.Error(1)
}

// Store mocks the Store method of UserRepository
func (m *MockUserRepository) Store(user *domain.User) error {
	args := m.Called(user)
	return args.Error(0)
}

func TestUserService_GetUserByID(t *testing.T) {
	tests := []struct {
		name          string
		userID        int64
		expectedUser  *domain.User
		expectedError error
	}{
		{
			name:   "successful retrieval",
			userID: 1,
			expectedUser: &domain.User{
				ID:        1,
				Name:      "Test User",
				Email:     "test@example.com",
				CreatedAt: time.Now(),
				UpdatedAt: time.Now(),
			},
			expectedError: nil,
		},
		{
			name:          "user not found",
			userID:        2,
			expectedUser:  nil,
			expectedError: domain.ErrNotFound,
		},
		{
			name:          "repository error",
			userID:        3,
			expectedUser:  nil,
			expectedError: errors.New("database error"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockRepo := new(MockUserRepository)

			var service *application.UserService
			app := fxtest.New(t,
				fx.Provide(func() domain.UserRepository { return mockRepo }),
				fx.Provide(application.NewUserService),
				fx.Populate(&service),
			)
			app.RequireStart().RequireStop()

			if tt.expectedError == domain.ErrNotFound {
				mockRepo.On("FindByID", mock.Anything, tt.userID).Return((*domain.User)(nil), domain.ErrNotFound).Once()
			} else if tt.expectedError != nil {
				mockRepo.On("FindByID", mock.Anything, tt.userID).Return((*domain.User)(nil), tt.expectedError).Once()
			} else {
				mockRepo.On("FindByID", mock.Anything, tt.userID).Return(tt.expectedUser, nil).Once()
			}

			user, err := service.GetUserByID(context.Background(), tt.userID)

			if tt.expectedError != nil {
				assert.Error(t, err)
				assert.True(t, errors.Is(err, tt.expectedError))
				assert.Nil(t, user)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectedUser, user)
			}

			mockRepo.AssertExpectations(t)
		})
	}
}
