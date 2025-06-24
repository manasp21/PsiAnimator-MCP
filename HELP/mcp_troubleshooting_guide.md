# MCP Server GitHub Actions Troubleshooting Guide

## Quick Diagnosis Commands

Before diving into specific fixes, run these commands locally to identify issues:

```bash
# For TypeScript projects
npm run build 2>&1 | tee build.log
npm run test 2>&1 | tee test.log
npm run lint 2>&1 | tee lint.log

# For Python projects
python -m build 2>&1 | tee build.log
python -m pytest 2>&1 | tee test.log
python -m flake8 . 2>&1 | tee lint.log

# Check MCP SDK compatibility
npm list @modelcontextprotocol/sdk  # TypeScript
pip show mcp  # Python
```

## Common Failure Categories & Solutions

### 1. Dependency Resolution Failures

#### **Error Pattern**: `npm ERR! Could not resolve dependency`
```
npm ERR! Could not resolve dependency:
npm ERR! peer @modelcontextprotocol/sdk@"^0.6.0" from psi-animator-mcp@1.0.0
```

**Root Cause**: Incompatible dependency versions or missing peer dependencies.

**Fix**:
```bash
# Clean and reinstall dependencies
rm -rf node_modules package-lock.json
npm cache clean --force
npm install

# For specific MCP SDK issues
npm install @modelcontextprotocol/sdk@latest
npm install --save-dev @types/node@latest

# Check for conflicts
npm ls --depth=0
```

**Prevention**: Update your `package.json`:
```json
{
  "engines": {
    "node": ">=18.0.0",
    "npm": ">=8.0.0"
  },
  "overrides": {
    "@modelcontextprotocol/sdk": "^0.6.0"
  }
}
```

---

#### **Error Pattern**: `ModuleNotFoundError: No module named 'mcp'`

**Root Cause**: Missing MCP Python SDK or virtual environment issues.

**Fix**:
```bash
# Ensure proper virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install MCP SDK
pip install mcp

# For development dependencies
pip install -e .[dev]

# Check installation
python -c "import mcp; print(mcp.__version__)"
```

### 2. Build Process Failures

#### **Error Pattern**: TypeScript compilation errors
```
src/index.ts(15,7): error TS2322: Type 'string' is not assignable to type 'number'
```

**Root Cause**: Type errors, missing declarations, or incorrect tsconfig.

**Fix**:
```bash
# Check TypeScript configuration
npx tsc --showConfig

# Fix common issues
npm install --save-dev @types/node @types/jest
npm install --save-dev typescript@latest

# Update tsconfig.json with MCP-specific settings
```

**Correct tsconfig.json** (see artifact above):
- Use `"target": "ES2022"` for modern Node.js
- Set `"moduleResolution": "Node"`
- Enable `"strict": true` for better type safety

---

#### **Error Pattern**: Missing build script
```
npm ERR! missing script: build
```

**Fix**: Add proper build scripts to `package.json`:
```json
{
  "scripts": {
    "build": "tsc -p .",
    "clean": "rimraf dist",
    "prebuild": "npm run clean",
    "start": "node dist/index.js"
  }
}
```

### 3. MCP SDK Integration Issues

#### **Error Pattern**: Protocol version mismatch
```
Error: MCP protocol version mismatch. Expected 0.6.0, got 0.5.2
```

**Root Cause**: Using incompatible MCP SDK versions.

**Fix**:
```bash
# Check current version
npm list @modelcontextprotocol/sdk

# Update to latest compatible version
npm install @modelcontextprotocol/sdk@latest

# For Python
pip install mcp --upgrade
```

**Code Fix**: Ensure proper MCP server initialization:
```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "psi-animator-mcp",
  version: "1.0.0",
}, {
  capabilities: {
    tools: {},
    resources: {},
    prompts: {}
  }
});
```

#### **Error Pattern**: Transport initialization failure
```
Error: Failed to initialize stdio transport
```

**Fix**: Correct transport setup:
```typescript
// Ensure proper async/await handling
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
```

### 4. Environment Setup Issues

#### **Error Pattern**: Node.js version incompatibility
```
Error: The engine "node" is incompatible with this module
```

**Fix**: Update GitHub Actions workflow:
```yaml
- name: Setup Node.js
  uses: actions/setup-node@v4
  with:
    node-version: '20'  # Use LTS version
    cache: 'npm'
```

**Fix**: Update `package.json` engines:
```json
{
  "engines": {
    "node": ">=18.0.0"
  }
}
```

#### **Error Pattern**: Python version issues
```
ERROR: Package 'mcp' requires a different Python: 3.8.0 not in '>=3.9'
```

**Fix**: Update Python version in workflow:
```yaml
- name: Setup Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.11'
```

### 5. Testing Failures

#### **Error Pattern**: Jest configuration issues
```
● Test suite failed to run
  Configuration error: Could not locate module @modelcontextprotocol/sdk
```

**Fix**: Update Jest configuration in `package.json`:
```json
{
  "jest": {
    "preset": "ts-jest",
    "testEnvironment": "node",
    "moduleNameMapping": {
      "^@modelcontextprotocol/(.*)$": "<rootDir>/node_modules/@modelcontextprotocol/$1"
    },
    "transform": {
      "^.+\\.tsx?$": "ts-jest"
    }
  }
}
```

#### **Error Pattern**: Async test timeouts
```
thrown: "Exceeded timeout of 5000 ms for a test"
```

**Fix**: Configure proper timeouts for MCP tests:
```typescript
describe('MCP Server', () => {
  beforeEach(() => {
    jest.setTimeout(30000); // 30 second timeout
  });

  test('should initialize server', async () => {
    // Your test code
  });
});
```

### 6. Linting and Code Quality Issues

#### **Error Pattern**: ESLint errors
```
✖ 15 problems (8 errors, 7 warnings)
  8 errors and 0 warnings potentially fixable with the `--fix` option.
```

**Fix**:
```bash
# Auto-fix issues
npx eslint . --fix

# Update ESLint configuration
npm install --save-dev @typescript-eslint/eslint-plugin@latest
```

**ESLint configuration** (add to `package.json`):
```json
{
  "eslintConfig": {
    "parser": "@typescript-eslint/parser",
    "plugins": ["@typescript-eslint"],
    "extends": [
      "eslint:recommended",
      "@typescript-eslint/recommended"
    ],
    "rules": {
      "@typescript-eslint/no-unused-vars": ["error", { "argsIgnorePattern": "^_" }]
    }
  }
}
```

### 7. Docker Build Failures

#### **Error Pattern**: Docker build context issues
```
ERROR: failed to solve: failed to read dockerfile
```

**Fix**: Create proper Dockerfile for MCP server:
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY dist ./dist
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

### 8. Security Scanning Issues

#### **Error Pattern**: Vulnerability warnings
```
found 3 high severity vulnerabilities
```

**Fix**:
```bash
# Audit and fix vulnerabilities
npm audit fix

# For unfixable issues, check if they affect MCP functionality
npm audit --audit-level=high

# Update dependencies
npm update
```

## Advanced Debugging Techniques

### Enable Debug Logging in CI

Add to your GitHub Actions:
```yaml
env:
  DEBUG: "*"
  NODE_ENV: development
  MCP_DEBUG: true
```

### Test MCP Server Locally

```bash
# Test server startup
node dist/index.js &
SERVER_PID=$!

# Give server time to initialize
sleep 2

# Test basic functionality
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}}}' | node dist/index.js

# Cleanup
kill $SERVER_PID
```

### Validate MCP Protocol Compliance

```typescript
// Add to your tests
import { validateMcpMessage } from '@modelcontextprotocol/sdk/types.js';

test('should handle valid MCP messages', async () => {
  const message = {
    jsonrpc: "2.0",
    id: 1,
    method: "tools/list",
    params: {}
  };
  
  const isValid = validateMcpMessage(message);
  expect(isValid).toBe(true);
});
```

## Prevention Strategies

### 1. Pre-commit Hooks

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-json
      - id: check-yaml

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11
```

### 2. Dependabot Configuration

Create `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10

  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
```

### 3. Automated Testing Strategy

```bash
# Local testing script
#!/bin/bash
set -e

echo "Running comprehensive MCP server tests..."

# TypeScript checks
if [ -f "tsconfig.json" ]; then
    echo "Type checking..."
    npx tsc --noEmit
fi

# Build
echo "Building..."
npm run build

# Tests
echo "Running tests..."
npm test

# Linting
echo "Linting..."
npm run lint

# Security
echo "Security scan..."
npm audit

# MCP validation
echo "Validating MCP server..."
timeout 10s node dist/index.js --version

echo "All checks passed! ✅"
```

## Emergency Recovery Steps

If your build is completely broken:

1. **Reset to last working commit**:
   ```bash
   git log --oneline -10
   git reset --hard <last-working-commit>
   ```

2. **Clean reinstall**:
   ```bash
   rm -rf node_modules package-lock.json dist/
   npm cache clean --force
   npm install
   ```

3. **Use minimal working configuration**:
   - Copy the provided `package.json`, `tsconfig.json`, or `pyproject.toml` templates
   - Start with basic MCP server example
   - Gradually add your custom functionality

4. **Test locally before pushing**:
   ```bash
   # Use the GitHub Actions locally
   npm install -g act
   act push
   ```

## Getting Help

- **MCP Documentation**: https://modelcontextprotocol.io/docs
- **TypeScript SDK**: https://github.com/modelcontextprotocol/typescript-sdk
- **Python SDK**: https://github.com/modelcontextprotocol/python-sdk
- **Community Discord**: https://discord.gg/anthropic

Remember: Most MCP server issues stem from dependency conflicts, incorrect configurations, or protocol version mismatches. The systematic approach above should resolve 95% of common problems.