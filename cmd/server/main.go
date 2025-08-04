package main

import (
	"go-starter-template/internal/app"

	"go.uber.org/fx"
)

func main() {
	fx.New(
		app.Module,
	).Run()
}
