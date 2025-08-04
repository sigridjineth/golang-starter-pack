package infrastructure

import (
	"context"
	"go-starter-template/internal/domain"
	"sync"

	pkgerrors "github.com/pkg/errors"
)

// InMemoryUserRepository is a simple in-memory implementation of UserRepository for demonstration purposes.
type InMemoryUserRepository struct {
	mu    sync.RWMutex
	users map[int64]*domain.User
}

// NewInMemoryUserRepository creates a new in-memory user repository.
func NewInMemoryUserRepository() *InMemoryUserRepository {
	return &InMemoryUserRepository{
		users: make(map[int64]*domain.User),
	}
}

// FindByID finds a user by their ID.
func (r *InMemoryUserRepository) FindByID(ctx context.Context, id int64) (*domain.User, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	user, ok := r.users[id]
	if !ok {
		return nil, pkgerrors.WithStack(domain.ErrNotFound)
	}
	return user, nil
}

// Store saves a user.
func (r *InMemoryUserRepository) Store(user *domain.User) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.users[user.ID] = user
	return nil
}
