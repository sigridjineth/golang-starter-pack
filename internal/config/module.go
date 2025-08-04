package config

import (
	"go.uber.org/fx"
)

// Module exports the configuration module
var Module = fx.Module("config",
	fx.Provide(func() (*Config, error) {
		return LoadConfig("./configs") // Assuming config.yaml is in a 'configs' directory
	}),
)
