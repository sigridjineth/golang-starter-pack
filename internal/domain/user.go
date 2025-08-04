package domain

import (
	"context"
	"errors"
	"time"
)

var ErrNotFound = errors.New("user not found")

// User represents a user in the system.
type User struct {
	ID        int64     `json:"id"`
	Name      string    `json:"name"`
	Email     string    `json:"email"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// UserRepository defines the interface for user data storage.
// This allows us to decouple the application logic from the data access layer.
type UserRepository interface {
	FindByID(ctx context.Context, id int64) (*User, error)
	Store(user *User) error
}
