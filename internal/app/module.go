package app

import (
	"context"
	"net/http"

	"go-starter-template/internal/application"
	"go-starter-template/internal/config"
	"go-starter-template/internal/domain"
	"go-starter-template/internal/infrastructure"
	interfaces "go-starter-template/internal/interfaces/http"

	os "os"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"go.uber.org/fx"
)

// Module exports the application module
var Module = fx.Options(
	config.Module,
	fx.Provide(func(cfg *config.Config) *zerolog.Logger {
		zerolog.TimeFieldFormat = zerolog.TimeFormatUnix

		if cfg.Logger.Level == "debug" {
			log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
		}

		// Add default fields
		logger := log.With().
			Str("service", "go-starter-template").
			Str("environment", cfg.Logger.Level). // Using log level as environment for simplicity
			Logger()
		return &logger
	}),
	fx.Provide(
		// Note: We are binding the interface to the implementation.
		// Fx will know that when a component asks for a domain.UserRepository,
		// it should provide an *infrastructure.InMemoryUserRepository.
		func() domain.UserRepository {
			return infrastructure.NewInMemoryUserRepository()
		},
		application.NewUserService,
		fx.Annotate(
			application.NewUserService,
			fx.As(new(application.UserServiceInterface)),
		),
		interfaces.NewUserHandler,
		interfaces.NewRouter,
	),
	// Provide the HTTP server
	fx.Provide(func(cfg *config.Config, router *http.ServeMux) *http.Server {
		return &http.Server{
			Addr:         ":" + cfg.Server.Port,
			Handler:      router,
			ReadTimeout:  cfg.Server.ReadTimeout,
			WriteTimeout: cfg.Server.WriteTimeout,
			IdleTimeout:  cfg.Server.IdleTimeout,
		}
	}),
	// Invoke a function that starts the HTTP server.
	// Fx will inject the *http.Server and fx.Lifecycle dependencies.
	fx.Invoke(func(lc fx.Lifecycle, server *http.Server, logger *zerolog.Logger) {
		lc.Append(fx.Hook{
			OnStart: func(ctx context.Context) error {
				logger.Info().Msgf("HTTP server starting on %s", server.Addr)
				go func() {
					if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
						logger.Error().Err(err).Msg("HTTP server failed to start")
					}
				}()
				return nil
			},
			OnStop: func(ctx context.Context) error {
				logger.Info().Msg("Shutting down server...")
				return server.Shutdown(ctx)
			},
		})
	}),
)
