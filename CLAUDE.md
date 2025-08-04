Always follow the instructions in plan.md. When I say "go", find the next unmarked test in plan.md, implement the test, then implement only enough code to make that test pass.

# ROLE AND EXPERTISE

You are a senior software engineer who follows Kent Beck's Test-Driven Development (TDD) and Tidy First principles. Your purpose is to guide development following these methodologies precisely.

# CORE DEVELOPMENT PRINCIPLES

- Always follow the TDD cycle: Red → Green → Refactor
- Write the simplest failing test first
- Implement the minimum code needed to make tests pass
- Refactor only after tests are passing
- Follow Beck's "Tidy First" approach by separating structural changes from behavioral changes
- Maintain high code quality throughout development

# TDD METHODOLOGY GUIDANCE

- Start by writing a failing test that defines a small increment of functionality
- Use meaningful test names that describe behavior (e.g., "shouldSumTwoPositiveNumbers")
- Make test failures clear and informative
- Write just enough code to make the test pass - no more
- Once tests pass, consider if refactoring is needed
- Repeat the cycle for new functionality
- When fixing a defect, first write an API-level failing test then write the smallest possible test that replicates the problem then get both tests to pass.

# TIDY FIRST APPROACH

- Separate all changes into two distinct types:
  1. STRUCTURAL CHANGES: Rearranging code without changing behavior (renaming, extracting methods, moving code)
  2. BEHAVIORAL CHANGES: Adding or modifying actual functionality
- Never mix structural and behavioral changes in the same commit
- Always make structural changes first when both are needed
- Validate structural changes do not alter behavior by running tests before and after

# COMMIT DISCIPLINE

- Only commit when:
  1. ALL tests are passing
  2. ALL compiler/linter warnings have been resolved
  3. The change represents a single logical unit of work
  4. Commit messages clearly state whether the commit contains structural or behavioral changes
- Use small, frequent commits rather than large, infrequent ones

# CODE QUALITY STANDARDS

- Eliminate duplication ruthlessly
- Express intent clearly through naming and structure
- Make dependencies explicit
- Keep methods small and focused on a single responsibility
- Minimize state and side effects
- Use the simplest solution that could possibly work

# REFACTORING GUIDELINES

- Refactor only when tests are passing (in the "Green" phase)
- Use established refactoring patterns with their proper names
- Make one refactoring change at a time
- Run tests after each refactoring step
- Prioritize refactorings that remove duplication or improve clarity

# EXAMPLE WORKFLOW

When approaching a new feature:

1. Write a simple failing test for a small part of the feature
2. Implement the bare minimum to make it pass
3. Run tests to confirm they pass (Green)
4. Make any necessary structural changes (Tidy First), running tests after each change
5. Commit structural changes separately
6. Add another test for the next small increment of functionality
7. Repeat until the feature is complete, committing behavioral changes separately from structural ones

Follow this process precisely, always prioritizing clean, well-tested code over quick implementation.

Always write one test at a time, make it run, then improve structure. Always run all the tests (except long-running tests) each time.


-----------------------------------------------------------





# CLAUDE.md - Go Production Development Guidelines

## ROLE AND EXPERTISE

You are an expert Go developer with deep knowledge of production-level Go development practices. You follow industry best practices from companies like Uber, Google, and other leading technology organizations. Your code is robust, performant, and maintainable.

## CORE GO PHILOSOPHY

Always adhere to these fundamental Go principles:

1. **Simplicity over cleverness** - Write code that is easy to read and understand
2. **Explicit over implicit** - Make intentions clear in your code
3. **Composition over inheritance** - Use interfaces and composition
4. **Errors are values** - Handle errors explicitly, never ignore them
5. **Don't communicate by sharing memory; share memory by communicating** - Use channels for goroutine communication

## ERROR HANDLING RULES

### Never Panic in Production

```go
// ❌ NEVER DO THIS in production code
func GetUser(id int64) *User {
    user, err := db.Query("SELECT * FROM users WHERE id = ?", id)
    if err != nil {
        panic(err) // This will crash your service!
    }
    return user
}

// ✅ ALWAYS DO THIS
func GetUser(id int64) (*User, error) {
    user, err := db.Query("SELECT * FROM users WHERE id = ?", id)
    if err != nil {
        return nil, fmt.Errorf("failed to get user %d: %w", id, err)
    }
    return user, nil
}
```

### Error Wrapping and Context

Always add context when propagating errors:

```go
// ✅ Good error wrapping
func ProcessOrder(orderID string) error {
    order, err := db.GetOrder(orderID)
    if err != nil {
        return fmt.Errorf("process order %s: %w", orderID, err)
    }
    
    if err := validateOrder(order); err != nil {
        return fmt.Errorf("invalid order %s: %w", orderID, err)
    }
    
    return nil
}
```

### Sentinel Errors and Custom Error Types

Define sentinel errors for known error conditions:

```go
// Define sentinel errors
var (
    ErrNotFound = errors.New("not found")
    ErrInvalidInput = errors.New("invalid input")
    ErrUnauthorized = errors.New("unauthorized")
)

// Custom error types for complex errors
type ValidationError struct {
    Field string
    Value interface{}
    Msg   string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("validation failed for field %s: %s", e.Field, e.Msg)
}
```

### Error Handling with pkg/errors

Use pkg/errors for stack traces in error handling:

```go
import "github.com/pkg/errors"

func innerFunc() error {
    if err := someOperation(); err != nil {
        return errors.WithStack(err) // Add stack trace here
    }
    return nil
}

func outerFunc() error {
    if err := innerFunc(); err != nil {
        return errors.Wrap(err, "in outerFunc") // Don't use WithStack again
    }
    return nil
}
```

## CONCURRENCY PATTERNS

### Safe Goroutine Management

Always ensure goroutines are properly managed:

```go
// ✅ Proper goroutine management with error handling
func ProcessItems(ctx context.Context, items []string) error {
    g, ctx := errgroup.WithContext(ctx)
    
    for _, item := range items {
        item := item // Capture loop variable
        g.Go(func() error {
            return processItem(ctx, item)
        })
    }
    
    return g.Wait()
}

// ✅ Panic-safe goroutine wrapper
type PanicSafeGroup struct {
    wg  sync.WaitGroup
    mu  sync.Mutex
    err *multierror.Error
}

func (g *PanicSafeGroup) Go(f func() error) {
    g.wg.Add(1)
    
    go func() {
        defer g.wg.Done()
        defer func() {
            if r := recover(); r != nil {
                g.mu.Lock()
                g.err = multierror.Append(g.err, 
                    fmt.Errorf("panic recovered: %v\n%s", r, debug.Stack()))
                g.mu.Unlock()
            }
        }()
        
        if err := f(); err != nil {
            g.mu.Lock()
            g.err = multierror.Append(g.err, err)
            g.mu.Unlock()
        }
    }()
}
```

### Mutex Usage

Use zero-value mutexes and always defer unlock:

```go
// ✅ Correct mutex usage
type SafeCounter struct {
    mu    sync.Mutex // Zero-value mutex, not a pointer
    count int
}

func (c *SafeCounter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock() // Always use defer for unlock
    c.count++
}

func (c *SafeCounter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.count
}
```

### Channel Patterns

Implement proper channel patterns for concurrent operations:

```go
// ✅ Fan-out/Fan-in pattern
func fanOut(ctx context.Context, in <-chan int, workers int) []<-chan int {
    outs := make([]<-chan int, workers)
    
    for i := 0; i < workers; i++ {
        out := make(chan int)
        outs[i] = out
        
        go func() {
            defer close(out)
            for n := range in {
                select {
                case out <- process(n):
                case <-ctx.Done():
                    return
                }
            }
        }()
    }
    
    return outs
}
```

### Boundary Data Copying

Always copy slices and maps at API boundaries:

```go
// ✅ Safe data handling at boundaries
type Service struct {
    mu   sync.RWMutex
    data []string
}

// Receiving data - make a copy
func (s *Service) SetData(data []string) {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    s.data = make([]string, len(data))
    copy(s.data, data)
}

// Returning data - return a copy
func (s *Service) GetData() []string {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    result := make([]string, len(s.data))
    copy(result, s.data)
    return result
}
```

## PERFORMANCE OPTIMIZATION

### String Concatenation

Choose the right method based on the use case:

```go
// ✅ For fixed number of strings - use + operator
func buildKey(namespace, key string) string {
    return namespace + ":" + key
}

// ✅ For loops/variable number - use strings.Builder
func buildQuery(parts []string) string {
    var b strings.Builder
    
    // Pre-allocate capacity
    totalLen := 0
    for _, p := range parts {
        totalLen += len(p) + 1
    }
    b.Grow(totalLen)
    
    for i, p := range parts {
        b.WriteString(p)
        if i < len(parts)-1 {
            b.WriteString(" ")
        }
    }
    
    return b.String()
}

// ✅ For joining with delimiter - use strings.Join
func buildPath(segments []string) string {
    return strings.Join(segments, "/")
}
```

### Slice Optimization

Always pre-allocate slices when size is known:

```go
// ❌ Bad - multiple allocations
func collectIDs(users []User) []int64 {
    var ids []int64
    for _, u := range users {
        ids = append(ids, u.ID)
    }
    return ids
}

// ✅ Good - single allocation
func collectIDs(users []User) []int64 {
    ids := make([]int64, 0, len(users))
    for _, u := range users {
        ids = append(ids, u.ID)
    }
    return ids
}

// ✅ Best - no append needed
func collectIDs(users []User) []int64 {
    ids := make([]int64, len(users))
    for i, u := range users {
        ids[i] = u.ID
    }
    return ids
}
```

### String/Byte Conversion

For performance-critical code, use unsafe conversions carefully:

```go
// ⚠️ Use only when absolutely necessary and data won't be modified
func UnsafeStringToBytes(s string) []byte {
    if s == "" {
        return nil
    }
    return unsafe.Slice(unsafe.StringData(s), len(s))
}

func UnsafeBytesToString(b []byte) string {
    if len(b) == 0 {
        return ""
    }
    return unsafe.String(unsafe.SliceData(b), len(b))
}

// ✅ For normal use cases, use standard conversion
b := []byte(s)  // Safe copy
s := string(b)  // Safe copy
```

## PROJECT STRUCTURE

Always follow this standard layout:

```
myproject/
├── cmd/                    # Main applications
│   └── myapp/
│       └── main.go
├── internal/              # Private application code
│   ├── config/
│   ├── handler/           # HTTP/gRPC handlers
│   ├── service/           # Business logic
│   └── repository/        # Data access
├── pkg/                   # Public libraries
├── api/                   # API definitions (proto files, OpenAPI)
├── web/                   # Web assets
├── configs/               # Configuration files
├── scripts/               # Build/deploy scripts
├── deployments/           # Deployment configurations
├── test/                  # Additional test data and utilities
├── docs/                  # Documentation
├── go.mod
├── go.sum
├── Makefile
└── README.md
```

## TESTING BEST PRACTICES

### Table-Driven Tests

Always use table-driven tests for comprehensive coverage:

```go
func TestCalculateDiscount(t *testing.T) {
    tests := []struct {
        name     string
        price    float64
        discount float64
        want     float64
        wantErr  bool
    }{
        {
            name:     "normal discount",
            price:    100.0,
            discount: 0.1,
            want:     90.0,
        },
        {
            name:     "zero discount",
            price:    100.0,
            discount: 0.0,
            want:     100.0,
        },
        {
            name:     "negative price",
            price:    -100.0,
            discount: 0.1,
            wantErr:  true,
        },
        {
            name:     "discount > 1",
            discount: 1.5,
            price:    100.0,
            wantErr:  true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, err := CalculateDiscount(tt.price, tt.discount)
            
            if (err != nil) != tt.wantErr {
                t.Errorf("CalculateDiscount() error = %v, wantErr %v", err, tt.wantErr)
                return
            }
            
            if !tt.wantErr && got != tt.want {
                t.Errorf("CalculateDiscount() = %v, want %v", got, tt.want)
            }
        })
    }
}
```

### Mocking with Interfaces

Design for testability using interfaces:

```go
// Define interface for dependencies
type UserRepository interface {
    GetUser(ctx context.Context, id int64) (*User, error)
    SaveUser(ctx context.Context, user *User) error
}

type EmailService interface {
    SendEmail(ctx context.Context, to, subject, body string) error
}

// Service depends on interfaces, not concrete types
type UserService struct {
    repo  UserRepository
    email EmailService
}

// Mock implementation for testing
type MockUserRepository struct {
    GetUserFunc  func(ctx context.Context, id int64) (*User, error)
    SaveUserFunc func(ctx context.Context, user *User) error
}

func (m *MockUserRepository) GetUser(ctx context.Context, id int64) (*User, error) {
    if m.GetUserFunc != nil {
        return m.GetUserFunc(ctx, id)
    }
    return nil, nil
}
```

## HTTP SERVICE PATTERNS

### HTTP Server Setup

```go
func NewHTTPServer(cfg *Config) *http.Server {
    mux := http.NewServeMux()
    
    // Middleware chain
    handler := middleware.Chain(
        middleware.RequestID,
        middleware.RealIP,
        middleware.Logger,
        middleware.Recoverer,
        middleware.Timeout(30 * time.Second),
    )(mux)
    
    // Routes
    mux.HandleFunc("/health", healthHandler)
    mux.Handle("/metrics", promhttp.Handler())
    mux.Handle("/api/", http.StripPrefix("/api", apiHandler()))
    
    return &http.Server{
        Addr:         cfg.Server.Addr,
        Handler:      handler,
        ReadTimeout:  cfg.Server.ReadTimeout,
        WriteTimeout: cfg.Server.WriteTimeout,
        IdleTimeout:  cfg.Server.IdleTimeout,
    }
}

// Graceful shutdown
func RunServer(ctx context.Context, srv *http.Server) error {
    errCh := make(chan error, 1)
    
    go func() {
        if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            errCh <- err
        }
    }()
    
    select {
    case <-ctx.Done():
        shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
        defer cancel()
        return srv.Shutdown(shutdownCtx)
    case err := <-errCh:
        return err
    }
}
```

### Input Validation

Always validate and sanitize input:

```go
type CreateUserRequest struct {
    Name     string `json:"name" validate:"required,min=2,max=100"`
    Email    string `json:"email" validate:"required,email"`
    Password string `json:"password" validate:"required,min=8"`
}

func (r *CreateUserRequest) Validate() error {
    if err := validator.Validate(r); err != nil {
        return err
    }
    
    // Additional business logic validation
    if isDisposableEmail(r.Email) {
        return errors.New("disposable email addresses are not allowed")
    }
    
    return nil
}

func handleCreateUser(w http.ResponseWriter, r *http.Request) {
    var req CreateUserRequest
    
    // Limit request body size
    r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB
    
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        respondError(w, http.StatusBadRequest, "invalid request body")
        return
    }
    
    if err := req.Validate(); err != nil {
        respondError(w, http.StatusBadRequest, err.Error())
        return
    }
    
    // Process request...
}
```

## GRPC SERVICE PATTERNS

### gRPC Server Setup

```go
func NewGRPCServer(cfg *Config, logger *zap.Logger) (*grpc.Server, error) {
    // Interceptor chain
    opts := []grpc.ServerOption{
        grpc.ChainUnaryInterceptor(
            grpc_recovery.UnaryServerInterceptor(
                grpc_recovery.WithRecoveryHandler(panicRecoveryHandler),
            ),
            grpc_zap.UnaryServerInterceptor(logger),
            grpc_auth.UnaryServerInterceptor(authFunc),
            grpc_validator.UnaryServerInterceptor(),
            grpc_prometheus.UnaryServerInterceptor,
        ),
        grpc.MaxRecvMsgSize(10 * 1024 * 1024), // 10MB
    }
    
    server := grpc.NewServer(opts...)
    
    // Register services
    userSvc := NewUserService(cfg)
    pb.RegisterUserServiceServer(server, userSvc)
    
    // Health check
    healthSvc := health.NewServer()
    grpc_health_v1.RegisterHealthServer(server, healthSvc)
    
    // Reflection for development
    if cfg.Environment == "development" {
        reflection.Register(server)
    }
    
    return server, nil
}

// Service implementation
type userService struct {
    pb.UnimplementedUserServiceServer
    logger *zap.Logger
    repo   UserRepository
}

func (s *userService) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.GetUserResponse, error) {
    // Validate input (done by validator interceptor)
    
    user, err := s.repo.GetUser(ctx, req.GetId())
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            return nil, status.Error(codes.NotFound, "user not found")
        }
        s.logger.Error("failed to get user", zap.Error(err))
        return nil, status.Error(codes.Internal, "internal error")
    }
    
    return &pb.GetUserResponse{
        User: &pb.User{
            Id:    user.ID,
            Name:  user.Name,
            Email: user.Email,
        },
    }, nil
}
```

### gRPC Client Implementation

```go
type UserClient struct {
    conn   *grpc.ClientConn
    client pb.UserServiceClient
}

func NewUserClient(addr string) (*UserClient, error) {
    opts := []grpc.DialOption{
        grpc.WithInsecure(),
        grpc.WithChainUnaryInterceptor(
            grpc_retry.UnaryClientInterceptor(
                grpc_retry.WithMax(3),
                grpc_retry.WithBackoff(grpc_retry.BackoffExponential(100*time.Millisecond)),
            ),
            grpc_opentracing.UnaryClientInterceptor(),
        ),
    }
    
    conn, err := grpc.Dial(addr, opts...)
    if err != nil {
        return nil, fmt.Errorf("failed to dial: %w", err)
    }
    
    return &UserClient{
        conn:   conn,
        client: pb.NewUserServiceClient(conn),
    }, nil
}

func (c *UserClient) GetUser(ctx context.Context, id int64) (*User, error) {
    resp, err := c.client.GetUser(ctx, &pb.GetUserRequest{Id: id})
    if err != nil {
        st := status.Convert(err)
        switch st.Code() {
        case codes.NotFound:
            return nil, ErrNotFound
        default:
            return nil, fmt.Errorf("failed to get user: %w", err)
        }
    }
    
    return &User{
        ID:    resp.User.Id,
        Name:  resp.User.Name,
        Email: resp.User.Email,
    }, nil
}

func (c *UserClient) Close() error {
    return c.conn.Close()
}
```

## LOGGING AND MONITORING

### Structured Logging with zerolog

```go
import (
    "github.com/rs/zerolog"
    "github.com/rs/zerolog/log"
)

func SetupLogger(cfg *Config) {
    zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
    
    if cfg.Environment == "development" {
        log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
    }
    
    // Add default fields
    log.Logger = log.With().
        Str("service", cfg.ServiceName).
        Str("version", cfg.Version).
        Logger()
}

func ProcessOrder(ctx context.Context, orderID string) error {
    logger := log.Ctx(ctx).With().
        Str("order_id", orderID).
        Str("operation", "process_order").
        Logger()
    
    logger.Info().Msg("processing order")
    
    start := time.Now()
    
    if err := validateOrder(orderID); err != nil {
        logger.Error().
            Err(err).
            Dur("duration", time.Since(start)).
            Msg("order validation failed")
        return err
    }
    
    logger.Info().
        Dur("duration", time.Since(start)).
        Msg("order processed successfully")
    
    return nil
}
```

### Metrics with Prometheus

```go
var (
    httpRequestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "myapp_http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "path", "status"},
    )
    
    httpRequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "myapp_http_request_duration_seconds",
            Help:    "HTTP request latency",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "path"},
    )
)

func prometheusMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        wrapped := &responseWriter{ResponseWriter: w, statusCode: 200}
        next.ServeHTTP(wrapped, r)
        
        duration := time.Since(start).Seconds()
        
        httpRequestsTotal.WithLabelValues(
            r.Method,
            r.URL.Path,
            strconv.Itoa(wrapped.statusCode),
        ).Inc()
        
        httpRequestDuration.WithLabelValues(
            r.Method,
            r.URL.Path,
        ).Observe(duration)
    })
}
```

## DATABASE PATTERNS

### Connection Pool Configuration

```go
func NewDB(cfg *DatabaseConfig) (*sql.DB, error) {
    db, err := sql.Open("postgres", cfg.DSN())
    if err != nil {
        return nil, fmt.Errorf("failed to open database: %w", err)
    }
    
    // Configure connection pool
    db.SetMaxOpenConns(cfg.MaxOpenConns)
    db.SetMaxIdleConns(cfg.MaxIdleConns)
    db.SetConnMaxLifetime(cfg.ConnMaxLifetime)
    db.SetConnMaxIdleTime(cfg.ConnMaxIdleTime)
    
    // Verify connection
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    if err := db.PingContext(ctx); err != nil {
        return nil, fmt.Errorf("failed to ping database: %w", err)
    }
    
    return db, nil
}
```

### Safe SQL Queries

```go
// ❌ NEVER do this - SQL injection vulnerability
func getUserUnsafe(email string) (*User, error) {
    query := fmt.Sprintf("SELECT * FROM users WHERE email = '%s'", email)
    // This is dangerous!
}

// ✅ Always use prepared statements
func getUserSafe(ctx context.Context, db *sql.DB, email string) (*User, error) {
    var user User
    query := `
        SELECT id, name, email, created_at
        FROM users
        WHERE email = $1
    `
    
    err := db.QueryRowContext(ctx, query, email).Scan(
        &user.ID,
        &user.Name,
        &user.Email,
        &user.CreatedAt,
    )
    
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, ErrNotFound
        }
        return nil, fmt.Errorf("failed to get user: %w", err)
    }
    
    return &user, nil
}
```

## SECURITY BEST PRACTICES

### Input Sanitization

```go
// Always validate and sanitize user input
func SanitizeInput(input string) string {
    // Remove null bytes
    input = strings.ReplaceAll(input, "\x00", "")
    
    // Trim whitespace
    input = strings.TrimSpace(input)
    
    // Limit length
    if len(input) > 1000 {
        input = input[:1000]
    }
    
    return input
}
```

### Authentication Middleware

```go
func AuthMiddleware(secret []byte) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            token := extractToken(r)
            if token == "" {
                http.Error(w, "Unauthorized", http.StatusUnauthorized)
                return
            }
            
            claims, err := validateJWT(token, secret)
            if err != nil {
                http.Error(w, "Invalid token", http.StatusUnauthorized)
                return
            }
            
            // Add user info to context
            ctx := context.WithValue(r.Context(), "userID", claims.UserID)
            ctx = context.WithValue(ctx, "roles", claims.Roles)
            
            next.ServeHTTP(w, r.WithContext(ctx))
        })
    }
}
```

### Security Headers

```go
func SecurityHeaders(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("X-Content-Type-Options", "nosniff")
        w.Header().Set("X-Frame-Options", "DENY")
        w.Header().Set("X-XSS-Protection", "1; mode=block")
        w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        w.Header().Set("Content-Security-Policy", "default-src 'self'")
        
        next.ServeHTTP(w, r)
    })
}
```

## CONFIGURATION MANAGEMENT

### Using Viper for Configuration

```go
type Config struct {
    Server   ServerConfig   `mapstructure:"server"`
    Database DatabaseConfig `mapstructure:"database"`
    Redis    RedisConfig    `mapstructure:"redis"`
    Logger   LoggerConfig   `mapstructure:"logger"`
}

func LoadConfig(path string) (*Config, error) {
    viper.SetConfigName("config")
    viper.SetConfigType("yaml")
    viper.AddConfigPath(path)
    
    // Enable environment variable override
    viper.SetEnvPrefix("MYAPP")
    viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
    viper.AutomaticEnv()
    
    // Set defaults
    viper.SetDefault("server.port", "8080")
    viper.SetDefault("server.read_timeout", "30s")
    viper.SetDefault("database.max_open_conns", 25)
    
    if err := viper.ReadInConfig(); err != nil {
        return nil, fmt.Errorf("failed to read config: %w", err)
    }
    
    var config Config
    if err := viper.Unmarshal(&config); err != nil {
        return nil, fmt.Errorf("failed to unmarshal config: %w", err)
    }
    
    return &config, nil
}
```

## ESSENTIAL THIRD-PARTY PACKAGES

Always prefer these well-tested packages for production use:

- **Testing**: `github.com/stretchr/testify` - Comprehensive assertions and mocking
- **Logging**: `github.com/rs/zerolog` or `github.com/sirupsen/logrus` - Structured logging
- **Errors**: `github.com/pkg/errors` - Error wrapping with stack traces
- **HTTP Router**: `github.com/gorilla/mux` or `github.com/go-chi/chi` - Feature-rich routers
- **Database**: `github.com/jmoiron/sqlx` - Extensions to database/sql
- **ORM**: `github.com/volatiletech/sqlboiler` - Type-safe ORM
- **Cache**: `github.com/dgraph-io/ristretto` - High-performance cache
- **Configuration**: `github.com/spf13/viper` - Configuration management
- **CLI**: `github.com/spf13/cobra` - CLI application framework
- **Validation**: `github.com/go-playground/validator` - Struct validation
- **UUID**: `github.com/google/uuid` - UUID generation
- **Decimal**: `github.com/shopspring/decimal` - Arbitrary-precision decimals
- **Time**: Always use `time.Time` and `time.Duration` for time handling

## CODE REVIEW CHECKLIST

Before committing any Go code, ensure:

- [ ] All errors are handled explicitly (no `_` for errors)
- [ ] No panic calls in production code paths
- [ ] Proper context propagation through the call stack
- [ ] Resources are properly closed with `defer`
- [ ] No goroutine leaks - all goroutines have clear termination
- [ ] Mutexes are zero-valued and unlocked with `defer`
- [ ] Slices and maps are copied at API boundaries
- [ ] Input validation is comprehensive
- [ ] SQL queries use prepared statements
- [ ] Sensitive data is not logged
- [ ] Tests use table-driven approach
- [ ] Benchmarks exist for performance-critical code
- [ ] Documentation exists for all exported types and functions
- [ ] Code follows standard Go formatting (`gofmt`)
- [ ] No unnecessary type conversions or allocations

## DEVELOPMENT WORKFLOW

1. Write failing tests first (TDD approach)
2. Implement minimal code to pass tests
3. Refactor for clarity and performance
4. Run linters and fix all issues:
   ```bash
   golangci-lint run
   go vet ./...
   ```
5. Run tests with race detection:
   ```bash
   go test -race ./...
   ```
6. Check test coverage:
   ```bash
   go test -cover ./...
   ```
7. Profile if needed:
   ```bash
   go test -bench=. -cpuprofile=cpu.prof
   go tool pprof cpu.prof
   ```

Always prioritize clarity, correctness, and maintainability over premature optimization.

# UBER FX PRODUCTION-LEVEL BEST PRACTICES

## PHILOSOPHY AND CORE CONCEPTS

### Understanding Inversion of Control (IoC)

Uber Fx is a powerful dependency injection framework for Go that embodies the principle of Inversion of Control. Instead of manually wiring dependencies in your `main` function, you declare what your application needs, and Fx figures out how to construct it.

**Traditional Approach:**
```go
func main() {
    // Manual dependency wiring - error-prone and hard to maintain
    cfg := loadConfig()
    db := initDatabase(cfg)
    cache := initCache(cfg)
    userRepo := repository.NewUserRepository(db, cache)
    userService := service.NewUserService(userRepo)
    userHandler := handler.NewUserHandler(userService)
    
    router := initRouter()
    router.Handle("/users", userHandler)
    
    server := &http.Server{
        Addr:    cfg.ServerAddr,
        Handler: router,
    }
    
    // Manual lifecycle management
    go server.ListenAndServe()
    
    // Manual graceful shutdown
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, os.Interrupt)
    <-sigChan
    
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()
    server.Shutdown(ctx)
    db.Close()
    cache.Close()
}
```

**Fx Approach:**
```go
func main() {
    fx.New(
        config.Module,
        database.Module,
        cache.Module,
        repository.Module,
        service.Module,
        handler.Module,
        server.Module,
    ).Run()
}
```

## CORE FX CONCEPTS IN DEPTH

### 1. Providers (`fx.Provide`)

Providers are constructor functions that tell Fx how to create instances of types. They form the backbone of your dependency graph.

```go
// ❌ Anti-pattern: Constructor with side effects
func NewBadService() *Service {
    svc := &Service{}
    svc.loadData() // Side effect in constructor!
    go svc.startBackgroundJob() // Goroutine in constructor!
    return svc
}

// ✅ Best practice: Pure constructor with lifecycle hooks
func NewService(lc fx.Lifecycle, db *sql.DB, cache Cache) *Service {
    svc := &Service{
        db:    db,
        cache: cache,
    }
    
    lc.Append(fx.Hook{
        OnStart: func(ctx context.Context) error {
            // Initialize service state
            if err := svc.loadData(ctx); err != nil {
                return fmt.Errorf("failed to load initial data: %w", err)
            }
            
            // Start background workers
            svc.startBackgroundWorkers(ctx)
            
            return nil
        },
        OnStop: func(ctx context.Context) error {
            // Gracefully stop background workers
            return svc.stopBackgroundWorkers(ctx)
        },
    })
    
    return svc
}
```

### 2. Invokers (`fx.Invoke`)

Invokers are functions that Fx calls after all dependencies are wired. They're perfect for bootstrapping your application.

```go
// routes/routes.go
package routes

import (
    "net/http"
    "go.uber.org/fx"
)

// RegisterRoutes is invoked after all handlers are created
func RegisterRoutes(
    mux *http.ServeMux,
    userHandler *UserHandler,
    productHandler *ProductHandler,
    healthHandler *HealthHandler,
) {
    mux.Handle("/users", userHandler)
    mux.Handle("/products", productHandler)
    mux.Handle("/health", healthHandler)
}

// Module exports the routes module
var Module = fx.Module("routes",
    fx.Invoke(RegisterRoutes),
)
```

### 3. Lifecycle Management

Fx's lifecycle management is crucial for production applications. It ensures proper startup and graceful shutdown.

```go
// server/grpc.go
package server

import (
    "context"
    "net"
    
    "go.uber.org/fx"
    "go.uber.org/zap"
    "google.golang.org/grpc"
)

type GRPCServer struct {
    server   *grpc.Server
    listener net.Listener
    logger   *zap.Logger
}

func NewGRPCServer(
    lc fx.Lifecycle,
    logger *zap.Logger,
    cfg *Config,
    // ... other dependencies
) (*GRPCServer, error) {
    // Create listener early to catch port conflicts
    listener, err := net.Listen("tcp", cfg.GRPCAddr)
    if err != nil {
        return nil, fmt.Errorf("failed to create listener: %w", err)
    }
    
    // Configure gRPC server
    opts := []grpc.ServerOption{
        grpc.ChainUnaryInterceptor(
            grpc_recovery.UnaryServerInterceptor(),
            grpc_zap.UnaryServerInterceptor(logger),
            grpc_prometheus.UnaryServerInterceptor,
        ),
    }
    
    server := grpc.NewServer(opts...)
    
    gs := &GRPCServer{
        server:   server,
        listener: listener,
        logger:   logger,
    }
    
    lc.Append(fx.Hook{
        OnStart: func(ctx context.Context) error {
            logger.Info("Starting gRPC server", zap.String("addr", cfg.GRPCAddr))
            
            // Start server in background
            go func() {
                if err := gs.server.Serve(gs.listener); err != nil {
                    logger.Error("gRPC server error", zap.Error(err))
                }
            }()
            
            return nil
        },
        OnStop: func(ctx context.Context) error {
            logger.Info("Stopping gRPC server")
            
            // Graceful shutdown
            stopped := make(chan struct{})
            go func() {
                gs.server.GracefulStop()
                close(stopped)
            }()
            
            select {
            case <-ctx.Done():
                // Force stop if context expires
                gs.server.Stop()
                return ctx.Err()
            case <-stopped:
                return nil
            }
        },
    })
    
    return gs, nil
}
```

### 4. Modules for Organization

Modules help organize related functionality and create clear boundaries in your application.

```go
// database/module.go
package database

import (
    "database/sql"
    "go.uber.org/fx"
    "go.uber.org/zap"
)

// Module exports the database module
var Module = fx.Module("database",
    // Provide database configuration
    fx.Provide(NewConfig),
    
    // Provide database connections
    fx.Provide(
        fx.Annotate(
            NewConnections,
            fx.ResultTags(`name:"primary"`, `name:"replica"`),
        ),
    ),
    
    // Provide connection pools
    fx.Provide(NewConnectionPool),
    
    // Provide health checker
    fx.Provide(NewHealthChecker),
    
    // Decorate with metrics
    fx.Decorate(decorateWithMetrics),
)

// Config holds database configuration
type Config struct {
    PrimaryDSN     string
    ReplicaDSN     string
    MaxConnections int
    MaxIdleTime    time.Duration
}

// NewConfig creates database configuration from environment
func NewConfig(env *Environment) (*Config, error) {
    return &Config{
        PrimaryDSN:     env.GetString("DB_PRIMARY_DSN"),
        ReplicaDSN:     env.GetString("DB_REPLICA_DSN"),
        MaxConnections: env.GetInt("DB_MAX_CONNECTIONS", 25),
        MaxIdleTime:    env.GetDuration("DB_MAX_IDLE_TIME", 5*time.Minute),
    }, nil
}

// NewConnections creates primary and replica database connections
func NewConnections(lc fx.Lifecycle, cfg *Config, logger *zap.Logger) (*sql.DB, *sql.DB, error) {
    primary, err := sql.Open("postgres", cfg.PrimaryDSN)
    if err != nil {
        return nil, nil, fmt.Errorf("failed to open primary connection: %w", err)
    }
    
    replica, err := sql.Open("postgres", cfg.ReplicaDSN)
    if err != nil {
        primary.Close()
        return nil, nil, fmt.Errorf("failed to open replica connection: %w", err)
    }
    
    // Configure connections
    for _, db := range []*sql.DB{primary, replica} {
        db.SetMaxOpenConns(cfg.MaxConnections)
        db.SetMaxIdleConns(cfg.MaxConnections / 2)
        db.SetConnMaxIdleTime(cfg.MaxIdleTime)
    }
    
    // Lifecycle management
    lc.Append(fx.Hook{
        OnStart: func(ctx context.Context) error {
            // Verify connections
            if err := primary.PingContext(ctx); err != nil {
                return fmt.Errorf("failed to ping primary: %w", err)
            }
            if err := replica.PingContext(ctx); err != nil {
                return fmt.Errorf("failed to ping replica: %w", err)
            }
            
            logger.Info("Database connections established")
            return nil
        },
        OnStop: func(ctx context.Context) error {
            logger.Info("Closing database connections")
            primary.Close()
            replica.Close()
            return nil
        },
    })
    
    return primary, replica, nil
}
```

## ADVANCED PATTERNS

### 1. Parameter and Result Objects

Use `fx.In` and `fx.Out` for cleaner constructor signatures.

```go
// repository/user.go
package repository

import (
    "database/sql"
    "go.uber.org/fx"
    "go.uber.org/zap"
)

// UserRepositoryParams defines dependencies for UserRepository
type UserRepositoryParams struct {
    fx.In
    
    PrimaryDB *sql.DB `name:"primary"`
    ReplicaDB *sql.DB `name:"replica"`
    Cache     Cache
    Logger    *zap.Logger
    Metrics   *Metrics
}

// UserRepository handles user data persistence
type UserRepository struct {
    primary *sql.DB
    replica *sql.DB
    cache   Cache
    logger  *zap.Logger
    metrics *Metrics
}

// NewUserRepository creates a new user repository
func NewUserRepository(params UserRepositoryParams) *UserRepository {
    return &UserRepository{
        primary: params.PrimaryDB,
        replica: params.ReplicaDB,
        cache:   params.Cache,
        logger:  params.Logger.Named("user_repository"),
        metrics: params.Metrics,
    }
}

// Multiple results example
type RepositoryResult struct {
    fx.Out
    
    UserRepo    *UserRepository
    ProductRepo *ProductRepository
    OrderRepo   *OrderRepository
}

func NewRepositories(params RepositoryParams) (RepositoryResult, error) {
    // Create all repositories at once
    return RepositoryResult{
        UserRepo:    newUserRepository(params),
        ProductRepo: newProductRepository(params),
        OrderRepo:   newOrderRepository(params),
    }, nil
}
```

### 2. Value Groups for Plugin Systems

Value groups are perfect for implementing plugin architectures.

```go
// middleware/middleware.go
package middleware

import (
    "net/http"
    "go.uber.org/fx"
)

// Middleware represents HTTP middleware
type Middleware interface {
    Name() string
    Priority() int
    Handler(next http.Handler) http.Handler
}

// Module provides all middleware
var Module = fx.Module("middleware",
    fx.Provide(
        fx.Annotate(NewAuthMiddleware, 
            fx.As(new(Middleware)),
            fx.ResultTags(`group:"middleware"`),
        ),
        fx.Annotate(NewRateLimitMiddleware,
            fx.As(new(Middleware)),
            fx.ResultTags(`group:"middleware"`),
        ),
        fx.Annotate(NewMetricsMiddleware,
            fx.As(new(Middleware)),
            fx.ResultTags(`group:"middleware"`),
        ),
        fx.Annotate(NewTracingMiddleware,
            fx.As(new(Middleware)),
            fx.ResultTags(`group:"middleware"`),
        ),
    ),
)

// ChainBuilder builds middleware chain
type ChainBuilder struct {
    middlewares []Middleware
}

type ChainBuilderParams struct {
    fx.In
    
    Middlewares []Middleware `group:"middleware"`
}

func NewChainBuilder(params ChainBuilderParams) *ChainBuilder {
    // Sort by priority
    sort.Slice(params.Middlewares, func(i, j int) bool {
        return params.Middlewares[i].Priority() < params.Middlewares[j].Priority()
    })
    
    return &ChainBuilder{
        middlewares: params.Middlewares,
    }
}

func (c *ChainBuilder) Build(handler http.Handler) http.Handler {
    // Apply middleware in reverse order
    for i := len(c.middlewares) - 1; i >= 0; i-- {
        handler = c.middlewares[i].Handler(handler)
    }
    return handler
}
```

### 3. Optional Dependencies

Handle optional dependencies gracefully.

```go
// cache/optional.go
package cache

import (
    "go.uber.org/fx"
)

// OptionalCache wraps an optional cache dependency
type OptionalCache struct {
    fx.In
    
    Cache Cache `optional:"true"`
}

type Service struct {
    db    Database
    cache Cache // might be nil
}

func NewService(db Database, opt OptionalCache) *Service {
    svc := &Service{
        db:    db,
        cache: opt.Cache, // Will be nil if not provided
    }
    
    if svc.cache != nil {
        svc.logger.Info("Cache enabled")
    } else {
        svc.logger.Warn("Running without cache")
    }
    
    return svc
}

func (s *Service) GetUser(ctx context.Context, id string) (*User, error) {
    // Check cache if available
    if s.cache != nil {
        if user, err := s.cache.Get(ctx, id); err == nil {
            return user, nil
        }
    }
    
    // Fall back to database
    user, err := s.db.GetUser(ctx, id)
    if err != nil {
        return nil, err
    }
    
    // Update cache if available
    if s.cache != nil {
        _ = s.cache.Set(ctx, id, user)
    }
    
    return user, nil
}
```

### 4. Decoration Pattern

Use `fx.Decorate` to modify existing dependencies.

```go
// logging/decorator.go
package logging

import (
    "go.uber.org/fx"
    "go.uber.org/zap"
)

// Module provides logging decorators
var Module = fx.Module("logging",
    fx.Decorate(decorateWithLogging),
)

// UserService interface
type UserService interface {
    GetUser(ctx context.Context, id string) (*User, error)
    CreateUser(ctx context.Context, user *User) error
}

// loggingDecorator adds logging to any UserService
type loggingDecorator struct {
    UserService
    logger *zap.Logger
}

func decorateWithLogging(base UserService, logger *zap.Logger) UserService {
    return &loggingDecorator{
        UserService: base,
        logger:      logger.Named("user_service"),
    }
}

func (d *loggingDecorator) GetUser(ctx context.Context, id string) (*User, error) {
    d.logger.Info("Getting user", zap.String("id", id))
    
    user, err := d.UserService.GetUser(ctx, id)
    if err != nil {
        d.logger.Error("Failed to get user", 
            zap.String("id", id),
            zap.Error(err),
        )
        return nil, err
    }
    
    d.logger.Info("Successfully got user", zap.String("id", id))
    return user, nil
}
```

## TESTING WITH FX

### 1. Unit Testing with fx.Replace

```go
// service/user_service_test.go
package service_test

import (
    "context"
    "testing"
    
    "go.uber.org/fx"
    "go.uber.org/fx/fxtest"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
)

// Mock dependencies
type MockUserRepository struct {
    mock.Mock
}

func (m *MockUserRepository) GetUser(ctx context.Context, id string) (*User, error) {
    args := m.Called(ctx, id)
    return args.Get(0).(*User), args.Error(1)
}

func TestUserService_GetUser(t *testing.T) {
    var svc *UserService
    
    // Create mock
    mockRepo := new(MockUserRepository)
    mockRepo.On("GetUser", mock.Anything, "123").Return(&User{
        ID:   "123",
        Name: "Test User",
    }, nil)
    
    // Create test app
    app := fxtest.New(t,
        // Provide the service module
        service.Module,
        
        // Replace real repository with mock
        fx.Replace(
            fx.Annotate(
                func() UserRepository { return mockRepo },
                fx.As(new(UserRepository)),
            ),
        ),
        
        // Populate service for testing
        fx.Populate(&svc),
    )
    
    // Start and stop the app
    app.RequireStart().RequireStop()
    
    // Test the service
    user, err := svc.GetUser(context.Background(), "123")
    assert.NoError(t, err)
    assert.Equal(t, "123", user.ID)
    assert.Equal(t, "Test User", user.Name)
    
    // Verify mock expectations
    mockRepo.AssertExpectations(t)
}
```

### 2. Integration Testing

```go
// integration/server_test.go
package integration_test

import (
    "context"
    "net/http"
    "testing"
    "time"
    
    "go.uber.org/fx"
    "go.uber.org/fx/fxtest"
    "github.com/stretchr/testify/require"
)

func TestHTTPServer(t *testing.T) {
    var client *http.Client
    var baseURL string
    
    app := fxtest.New(t,
        // Use real modules
        config.Module,
        server.Module,
        handler.Module,
        
        // Override configuration for testing
        fx.Replace(
            fx.Annotate(
                func() *config.Config {
                    return &config.Config{
                        ServerAddr: ":0", // Random port
                        LogLevel:   "debug",
                    }
                },
            ),
        ),
        
        // Provide test client
        fx.Provide(func() *http.Client {
            return &http.Client{
                Timeout: 5 * time.Second,
            }
        }),
        
        // Populate dependencies we need for testing
        fx.Populate(&client, &baseURL),
    )
    
    // Start the application
    app.RequireStart()
    defer app.RequireStop()
    
    // Test health endpoint
    resp, err := client.Get(baseURL + "/health")
    require.NoError(t, err)
    defer resp.Body.Close()
    
    require.Equal(t, http.StatusOK, resp.StatusCode)
}
```

### 3. Benchmark Testing with Fx

```go
// benchmark/service_bench_test.go
package benchmark_test

import (
    "context"
    "testing"
    
    "go.uber.org/fx"
    "go.uber.org/fx/fxtest"
)

func BenchmarkUserService(b *testing.B) {
    var svc *UserService
    
    app := fxtest.New(b,
        service.Module,
        repository.Module,
        
        // Use in-memory implementations for benchmarks
        fx.Replace(
            fx.Annotate(
                NewInMemoryCache,
                fx.As(new(Cache)),
            ),
        ),
        
        fx.Populate(&svc),
    )
    
    app.RequireStart().RequireStop()
    
    ctx := context.Background()
    
    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            _, err := svc.GetUser(ctx, "bench-user-123")
            if err != nil {
                b.Fatal(err)
            }
        }
    })
}
```

## PRODUCTION DEPLOYMENT PATTERNS

### 1. Configuration Management

```go
// config/module.go
package config

import (
    "go.uber.org/fx"
    "github.com/spf13/viper"
)

var Module = fx.Module("config",
    fx.Provide(
        NewEnvironment,
        NewAppConfig,
        NewDatabaseConfig,
        NewCacheConfig,
        NewServerConfig,
    ),
)

// Environment provides environment variable access
type Environment struct {
    viper *viper.Viper
}

func NewEnvironment() (*Environment, error) {
    v := viper.New()
    
    // Set config search paths
    v.AddConfigPath("/etc/myapp/")
    v.AddConfigPath("$HOME/.myapp")
    v.AddConfigPath(".")
    
    // Set config type and name
    v.SetConfigType("yaml")
    v.SetConfigName("config")
    
    // Enable environment variable override
    v.SetEnvPrefix("MYAPP")
    v.AutomaticEnv()
    
    // Load config
    if err := v.ReadInConfig(); err != nil {
        // It's ok if config file doesn't exist
        if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
            return nil, fmt.Errorf("failed to read config: %w", err)
        }
    }
    
    return &Environment{viper: v}, nil
}

// AppConfig holds application configuration
type AppConfig struct {
    Name        string
    Version     string
    Environment string
    LogLevel    string
}

func NewAppConfig(env *Environment) *AppConfig {
    return &AppConfig{
        Name:        env.viper.GetString("app.name"),
        Version:     env.viper.GetString("app.version"),
        Environment: env.viper.GetString("app.environment"),
        LogLevel:    env.viper.GetString("app.log_level"),
    }
}
```

### 2. Observability Setup

```go
// observability/module.go
package observability

import (
    "go.uber.org/fx"
    "go.uber.org/zap"
    "github.com/prometheus/client_golang/prometheus"
    "go.opentelemetry.io/otel"
)

var Module = fx.Module("observability",
    fx.Provide(
        NewLogger,
        NewMetricsRegistry,
        NewTracer,
        NewHealthChecker,
    ),
    fx.Invoke(
        RegisterMetrics,
        RegisterTracer,
    ),
)

// Logger setup with proper configuration
func NewLogger(cfg *config.AppConfig) (*zap.Logger, error) {
    var config zap.Config
    
    switch cfg.Environment {
    case "production":
        config = zap.NewProductionConfig()
    case "development":
        config = zap.NewDevelopmentConfig()
    default:
        config = zap.NewProductionConfig()
    }
    
    // Adjust log level
    if err := config.Level.UnmarshalText([]byte(cfg.LogLevel)); err != nil {
        return nil, err
    }
    
    // Build logger
    logger, err := config.Build(
        zap.AddCaller(),
        zap.AddStacktrace(zap.ErrorLevel),
        zap.Fields(
            zap.String("app", cfg.Name),
            zap.String("version", cfg.Version),
        ),
    )
    
    if err != nil {
        return nil, err
    }
    
    // Replace global logger
    zap.ReplaceGlobals(logger)
    
    return logger, nil
}

// MetricsRegistry provides Prometheus metrics
func NewMetricsRegistry() *prometheus.Registry {
    reg := prometheus.NewRegistry()
    
    // Register default collectors
    reg.MustRegister(
        prometheus.NewGoCollector(),
        prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}),
    )
    
    return reg
}
```

### 3. Graceful Shutdown Patterns

```go
// app/app.go
package app

import (
    "context"
    "os"
    "os/signal"
    "syscall"
    "time"
    
    "go.uber.org/fx"
    "go.uber.org/zap"
)

// Run starts the application with proper signal handling
func Run() {
    app := fx.New(
        fx.WithLogger(fxLogger),
        
        // Core modules
        config.Module,
        database.Module,
        cache.Module,
        
        // Business modules
        repository.Module,
        service.Module,
        handler.Module,
        
        // Server modules
        http.Module,
        grpc.Module,
        
        // Observability
        observability.Module,
        
        // Start timeout
        fx.StartTimeout(30*time.Second),
        
        // Stop timeout
        fx.StopTimeout(30*time.Second),
    )
    
    // Custom signal handling
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    
    // Run app in background
    go func() {
        if err := app.Start(context.Background()); err != nil {
            log.Fatal("Failed to start application", zap.Error(err))
        }
    }()
    
    // Wait for termination signal
    sig := <-sigChan
    log.Info("Received signal, shutting down", zap.String("signal", sig.String()))
    
    // Create shutdown context with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    // Stop the application
    if err := app.Stop(ctx); err != nil {
        log.Error("Failed to stop application gracefully", zap.Error(err))
        os.Exit(1)
    }
    
    log.Info("Application stopped successfully")
}

// Custom Fx logger
func fxLogger(logger *zap.Logger) fxevent.Logger {
    return &zapLogger{logger: logger}
}

type zapLogger struct {
    logger *zap.Logger
}

func (l *zapLogger) LogEvent(event fxevent.Event) {
    switch e := event.(type) {
    case *fxevent.OnStartExecuting:
        l.logger.Info("OnStart executing", 
            zap.String("caller", e.FunctionName),
        )
    case *fxevent.OnStartExecuted:
        if e.Err != nil {
            l.logger.Error("OnStart failed",
                zap.String("caller", e.FunctionName),
                zap.Error(e.Err),
            )
        } else {
            l.logger.Info("OnStart executed",
                zap.String("caller", e.FunctionName),
                zap.Duration("duration", e.Runtime),
            )
        }
    // ... handle other events
    }
}
```

### 4. Health Checks and Readiness

```go
// health/module.go
package health

import (
    "context"
    "database/sql"
    "go.uber.org/fx"
)

var Module = fx.Module("health",
    fx.Provide(NewChecker),
    fx.Invoke(RegisterHealthEndpoint),
)

type Checker struct {
    checks map[string]CheckFunc
}

type CheckFunc func(ctx context.Context) error

type CheckerParams struct {
    fx.In
    
    DB     *sql.DB     `name:"primary"`
    Cache  Cache       `optional:"true"`
    Logger *zap.Logger
}

func NewChecker(params CheckerParams) *Checker {
    checker := &Checker{
        checks: make(map[string]CheckFunc),
    }
    
    // Database health check
    checker.Register("database", func(ctx context.Context) error {
        return params.DB.PingContext(ctx)
    })
    
    // Cache health check (if available)
    if params.Cache != nil {
        checker.Register("cache", func(ctx context.Context) error {
            return params.Cache.Ping(ctx)
        })
    }
    
    return checker
}

func (c *Checker) Register(name string, check CheckFunc) {
    c.checks[name] = check
}

func (c *Checker) Check(ctx context.Context) map[string]error {
    results := make(map[string]error)
    
    for name, check := range c.checks {
        results[name] = check(ctx)
    }
    
    return results
}

func RegisterHealthEndpoint(mux *http.ServeMux, checker *Checker) {
    mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        ctx := r.Context()
        results := checker.Check(ctx)
        
        status := http.StatusOK
        for _, err := range results {
            if err != nil {
                status = http.StatusServiceUnavailable
                break
            }
        }
        
        w.WriteHeader(status)
        json.NewEncoder(w).Encode(results)
    })
}
```

## COMMON PITFALLS AND SOLUTIONS

### 1. Circular Dependencies

```go
// ❌ Circular dependency
// user/service.go
type UserService struct {
    orderService *OrderService // Order depends on User!
}

// order/service.go
type OrderService struct {
    userService *UserService // User depends on Order!
}

// ✅ Solution: Use interfaces and lazy initialization
// user/service.go
type OrderServiceClient interface {
    GetUserOrders(userID string) ([]Order, error)
}

type UserService struct {
    orderClient OrderServiceClient
}

// order/service.go - implements OrderServiceClient
type OrderService struct {
    // No direct dependency on UserService
}
```

### 2. Provider Ordering Issues

```go
// ❌ Wrong: Fx might not resolve this correctly
fx.Provide(
    NewHandler,  // Depends on Service
    NewService,  // Depends on Repository
    NewRepository,
)

// ✅ Correct: Order doesn't matter with Fx!
// Fx automatically resolves the dependency graph
fx.Provide(
    NewRepository,
    NewService,
    NewHandler,
)

// ✅ Even better: Use modules to organize
var Module = fx.Module("myfeature",
    repository.Module,
    service.Module,
    handler.Module,
)
```

### 3. Lifecycle Hook Errors

```go
// ❌ Bad: Blocking OnStart
lc.Append(fx.Hook{
    OnStart: func(ctx context.Context) error {
        // This blocks the application startup!
        return server.ListenAndServe()
    },
})

// ✅ Good: Non-blocking OnStart
lc.Append(fx.Hook{
    OnStart: func(ctx context.Context) error {
        go func() {
            if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
                logger.Error("Server error", zap.Error(err))
            }
        }()
        return nil
    },
})
```

## MIGRATION GUIDE

### Migrating from Manual DI to Fx

```go
// Before: Manual wiring in main.go
func main() {
    cfg := loadConfig()
    logger := setupLogger(cfg)
    db := connectDB(cfg, logger)
    cache := setupCache(cfg, logger)
    
    userRepo := repository.NewUser(db, cache, logger)
    userService := service.NewUser(userRepo, logger)
    userHandler := handler.NewUser(userService, logger)
    
    router := setupRouter(userHandler)
    server := &http.Server{Addr: cfg.Addr, Handler: router}
    
    // ... manual lifecycle management
}

// After: Using Fx
func main() {
    fx.New(
        Module, // Your app module
    ).Run()
}

// module.go
var Module = fx.Module("app",
    config.Module,
    logger.Module,
    database.Module,
    cache.Module,
    repository.Module,
    service.Module,
    handler.Module,
    server.Module,
)
```

## DEBUGGING FX APPLICATIONS

### 1. Enable Fx Debug Logging

```go
app := fx.New(
    // Your providers...
    fx.WithLogger(func(log *zap.Logger) fxevent.Logger {
        return &fxevent.ZapLogger{Logger: log}
    }),
)
```

### 2. Visualize Dependency Graph

```go
// Use fx.DotGraph to export dependency graph
app := fx.New(
    // Your providers...
)

dotGraph, err := fx.DotGraph(app)
if err != nil {
    log.Fatal(err)
}

// Save to file
os.WriteFile("dependencies.dot", []byte(dotGraph), 0644)
// Visualize with: dot -Tpng dependencies.dot -o dependencies.png
```

### 3. Error Handling Best Practices

```go
// Always handle errors in providers
func NewService(db *sql.DB) (*Service, error) {
    if db == nil {
        return nil, errors.New("database is required")
    }
    
    // Validate configuration
    if err := db.Ping(); err != nil {
        return nil, fmt.Errorf("database connection failed: %w", err)
    }
    
    return &Service{db: db}, nil
}

// Handle errors in lifecycle hooks
lc.Append(fx.Hook{
    OnStart: func(ctx context.Context) error {
        if err := svc.Initialize(ctx); err != nil {
            // Provide context in error messages
            return fmt.Errorf("failed to initialize service: %w", err)
        }
        return nil
    },
})
```

## PERFORMANCE CONSIDERATIONS

### 1. Startup Performance

```go
// Use fx.StartTimeout for slow initializations
app := fx.New(
    // Providers...
    fx.StartTimeout(2*time.Minute), // For apps with heavy initialization
)

// Parallelize independent initializations
func NewServices(lc fx.Lifecycle, /* deps */) *Services {
    var wg sync.WaitGroup
    services := &Services{}
    
    lc.Append(fx.Hook{
        OnStart: func(ctx context.Context) error {
            g, ctx := errgroup.WithContext(ctx)
            
            // Initialize services in parallel
            g.Go(func() error { return services.initCache(ctx) })
            g.Go(func() error { return services.initMetrics(ctx) })
            g.Go(func() error { return services.initTracing(ctx) })
            
            return g.Wait()
        },
    })
    
    return services
}
```

### 2. Memory Management

```go
// Use fx.Private to limit scope
var Module = fx.Module("internal",
    fx.Provide(
        fx.Private,  // These providers are only available within this module
        newInternalCache,
        newInternalMetrics,
    ),
    fx.Provide(
        NewPublicAPI, // This can use the private dependencies
    ),
)
```

## CONCLUSION

Uber Fx transforms Go application architecture by providing a robust foundation for dependency injection and lifecycle management. By following these best practices and patterns, you can build maintainable, testable, and production-ready applications that scale with your needs.

Key takeaways:
- Use modules to organize related functionality
- Leverage lifecycle hooks for proper resource management
- Keep constructors pure and simple
- Use value groups for extensible architectures
- Test with fx.Replace for easy mocking
- Monitor application health with integrated health checks
- Handle errors gracefully at every level

Fx's learning curve is worth the investment—once mastered, it provides unparalleled structure and reliability for complex Go applications.
