#!/bin/sh
# init-matrix-users.sh
# 初始化 Matrix 系统用户

set -e

MATRIX_SERVER="${MATRIX_SERVER:-http://tuwunel:6167}"
MATRIX_DOMAIN="${MATRIX_DOMAIN:-matrix.semantic-router.local}"
REG_TOKEN="${MATRIX_REGISTRATION_TOKEN:-changeme}"

echo "🔧 Initializing Matrix users..."
echo "Server: ${MATRIX_SERVER}"
echo "Domain: ${MATRIX_DOMAIN}"

# 等待服务就绪
wait_for_server() {
    echo "Waiting for Matrix server..."
    for i in $(seq 1 30); do
        if curl -sf "${MATRIX_SERVER}/_matrix/client/versions" > /dev/null 2>&1; then
            echo "Matrix server is ready!"
            return 0
        fi
        echo "Attempt $i/30 - waiting..."
        sleep 2
    done
    echo "ERROR: Matrix server not ready after 60 seconds"
    exit 1
}

# 注册用户
register_user() {
    local username="$1"
    local password="${REG_TOKEN}"
    
    echo "Registering user: ${username}..."
    
    response=$(curl -sf -X POST "${MATRIX_SERVER}/_matrix/client/v3/register" \
        -H "Content-Type: application/json" \
        -d "{
            \"username\": \"${username}\",
            \"password\": \"${password}\",
            \"registration_token\": \"${REG_TOKEN}\",
            \"device_id\": \"semantic-router-${username}\",
            \"initial_device_display_name\": \"Semantic Router ${username}\"
        }" 2>&1 || echo "")
    
    if echo "$response" | grep -q '"user_id"'; then
        echo "✅ User ${username} registered successfully"
        echo "$response" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4
    elif echo "$response" | grep -q 'M_USER_IN_USE'; then
        echo "ℹ️  User ${username} already exists"
        # 尝试登录获取 token
        login_user "${username}" "${password}"
    else
        echo "⚠️  Failed to register ${username}: $response"
    fi
}

# 登录用户
login_user() {
    local username="$1"
    local password="$2"
    
    response=$(curl -sf -X POST "${MATRIX_SERVER}/_matrix/client/v3/login" \
        -H "Content-Type: application/json" \
        -d "{
            \"type\": \"m.login.password\",
            \"identifier\": {
                \"type\": \"m.id.user\",
                \"user\": \"${username}\"
            },
            \"password\": \"${password}\",
            \"device_id\": \"semantic-router-${username}\",
            \"initial_device_display_name\": \"Semantic Router ${username}\"
        }" 2>&1 || echo "")
    
    if echo "$response" | grep -q '"access_token"'; then
        echo "$response" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4
    fi
}

# 主流程
wait_for_server

echo ""
echo "📝 Creating system users..."

# 创建系统用户
SYSTEM_TOKEN=$(register_user "system")
ADMIN_TOKEN=$(register_user "admin")
LEADER_TOKEN=$(register_user "leader")

echo ""
echo "🎉 Matrix initialization complete!"
echo ""
echo "Generated tokens (save these for configuration):"
echo "================================================"
echo "SYSTEM_MATRIX_ACCESS_TOKEN=${SYSTEM_TOKEN}"
echo "ADMIN_MATRIX_ACCESS_TOKEN=${ADMIN_TOKEN}"
echo "LEADER_MATRIX_ACCESS_TOKEN=${LEADER_TOKEN}"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Save the tokens above to your .env file"
echo "2. Restart the dashboard and leader services"
echo "3. Access Element Web at http://localhost:8088"
