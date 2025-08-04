package infrastructure_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"go-starter-template/internal/domain"
	"go-starter-template/internal/infrastructure"

	"github.com/stretchr/testify/assert"
)

func TestInMemoryUserRepository_FindByID(t *testing.T) {
	repo := infrastructure.NewInMemoryUserRepository()

	// Prepare some test data
	user1 := &domain.User{
		ID:        1,
		Name:      "Test User 1",
		Email:     "test1@example.com",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	user2 := &domain.User{
		ID:        2,
		Name:      "Test User 2",
		Email:     "test2@example.com",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	assert.NoError(t, repo.Store(user1))
	assert.NoError(t, repo.Store(user2))

	tests := []struct {
		name          string
		userID        int64
		expectedUser  *domain.User
		expectedError error
	}{
		{
			name:          "user found",
			userID:        1,
			expectedUser:  user1,
			expectedError: nil,
		},
		{
			name:          "user not found",
			userID:        3,
			expectedUser:  nil,
			expectedError: domain.ErrNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			user, err := repo.FindByID(context.Background(), tt.userID)

			assert.True(t, errors.Is(err, tt.expectedError))
			assert.Equal(t, tt.expectedUser, user)
		})
	}
}

func TestInMemoryUserRepository_Store(t *testing.T) {
	repo := infrastructure.NewInMemoryUserRepository()

	user := &domain.User{
		ID:        1,
		Name:      "New User",
		Email:     "new@example.com",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	err := repo.Store(user)
	assert.NoError(t, err)

	foundUser, err := repo.FindByID(context.Background(), 1)
	assert.NoError(t, err)
	assert.Equal(t, user, foundUser)

	// Test update
	user.Name = "Updated User"
	err = repo.Store(user)
	assert.NoError(t, err)

	foundUser, err = repo.FindByID(context.Background(), 1)
	assert.NoError(t, err)
	assert.Equal(t, user, foundUser)
}
