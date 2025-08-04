package application

import (
	"context"
	"errors"
	"go-starter-template/internal/domain"

	pkgerrors "github.com/pkg/errors"
)

// UserServiceInterface defines the interface for user-related services.
type UserServiceInterface interface {
	GetUserByID(ctx context.Context, id int64) (*domain.User, error)
}

// UserService provides user-related services.
type UserService struct {
	repo domain.UserRepository
}

// NewUserService creates a new UserService.
func NewUserService(repo domain.UserRepository) *UserService {
	return &UserService{repo: repo}
}

// GetUserByID retrieves a user by their ID.
func (s *UserService) GetUserByID(ctx context.Context, id int64) (*domain.User, error) {
	user, err := s.repo.FindByID(ctx, id)
	if err != nil {
		if errors.Is(err, domain.ErrNotFound) {
			return nil, domain.ErrNotFound
		}
		return nil, pkgerrors.Wrapf(err, "failed to get user by ID %d", id)
	}
	return user, nil
}
