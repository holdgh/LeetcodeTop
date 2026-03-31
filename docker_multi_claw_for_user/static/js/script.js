/**
 * CoPaw 多用户实例管理系统 - 前端交互脚本
 * 版本: 1.1.0
 * 更新: 2026-03-30
 */

// 等待 Vue 和 Element Plus 加载完成
(function() {
    'use strict';
    
    // 检查依赖是否加载
    function checkDependencies() {
        if (typeof Vue === 'undefined') {
            console.error('Vue 未加载');
            return false;
        }
        if (typeof ElementPlus === 'undefined') {
            console.error('Element Plus 未加载');
            return false;
        }
        return true;
    }
    
    // 初始化应用
    function initApp() {
        if (!checkDependencies()) {
            document.getElementById('loading').innerHTML = '<div style="text-align: center; color: white;"><div style="font-size: 18px; margin-bottom: 16px;">⚠️ 加载失败</div><div style="font-size: 14px; opacity: 0.8;">请检查网络连接后刷新页面</div></div>';
            return;
        }
        
        const { createApp, ref, reactive, onMounted } = Vue;
        
        const app = createApp({
            setup() {
                // 响应式数据
                const activeTab = ref('1');
                const loaded = ref(false);
                const message = ref(null);
                
                const userForm = reactive({
                    userId: '',
                    userName: ''
                });
                
                const backupForm = reactive({
                    userId: ''
                });
                
                const metricsForm = reactive({
                    userId: ''
                });
                
                const loading = reactive({
                    register: false,
                    login: false,
                    logout: false,
                    delete: false,
                    list: false,
                    listBackups: false,
                    createBackup: false,
                    metrics: false
                });
                
                const instanceList = ref([]);
                const backupList = ref([]);
                const metricsData = ref(null);
                const recentOperations = ref([]);
                
                const stats = reactive({
                    totalInstances: 0,
                    runningInstances: 0,
                    stoppedInstances: 0
                });
                
                const apiList = ref([
                    { method: 'POST', path: '/admin/users/{user_id}/register', description: '用户注册 - 创建CoPaw容器实例' },
                    { method: 'POST', path: '/admin/users/{user_id}/login', description: '用户登录 - 启动用户容器' },
                    { method: 'POST', path: '/admin/users/{user_id}/logout', description: '用户登出 - 停止用户容器' },
                    { method: 'DELETE', path: '/admin/users/{user_id}', description: '删除用户 - 删除容器和数据' },
                    { method: 'GET', path: '/admin/instances', description: '获取所有实例列表' },
                    { method: 'GET', path: '/admin/backups/{user_id}', description: '获取用户备份列表' },
                    { method: 'POST', path: '/admin/backups/{user_id}/create', description: '创建用户数据备份' },
                    { method: 'GET', path: '/admin/metrics/{user_id}', description: '获取用户实例资源指标' },
                    { method: 'GET', path: '/health', description: '系统健康检查' }
                ]);
                
                // 方法
                const handleMenuSelect = (index) => {
                    activeTab.value = index;
                    if (index === '2') {
                        listInstances();
                    }
                };
                
                const showMessage = (title, content, type) => {
                    message.value = { title, content, type };
                    setTimeout(() => {
                        message.value = null;
                    }, 5000);
                };
                
                const addOperation = (action, userId, status, detail) => {
                    recentOperations.value.unshift({
                        time: formatTime(new Date()),
                        action,
                        userId,
                        status,
                        detail
                    });
                    if (recentOperations.value.length > 10) {
                        recentOperations.value.pop();
                    }
                };
                
                const updateStats = () => {
                    stats.totalInstances = instanceList.value.length;
                    stats.runningInstances = instanceList.value.filter(i => i.status === 'running').length;
                    stats.stoppedInstances = instanceList.value.filter(i => i.status === 'stopped').length;
                };
                
                const formatTime = (time) => {
                    if (!time) return '-';
                    const date = new Date(time);
                    const year = date.getFullYear();
                    const month = String(date.getMonth() + 1).padStart(2, '0');
                    const day = String(date.getDate()).padStart(2, '0');
                    const hours = String(date.getHours()).padStart(2, '0');
                    const minutes = String(date.getMinutes()).padStart(2, '0');
                    const seconds = String(date.getSeconds()).padStart(2, '0');
                    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
                };
                
                const formatSize = (bytes) => {
                    if (!bytes || bytes === 0) return '0 B';
                    const k = 1024;
                    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
                    const i = Math.floor(Math.log(bytes) / Math.log(k));
                    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
                };
                
                const getProgressColor = (value) => {
                    if (value < 50) return '#67c23a';
                    if (value < 80) return '#e6a23c';
                    return '#f56c6c';
                };
                
                const getMethodType = (method) => {
                    const types = {
                        'GET': 'success',
                        'POST': '',
                        'PUT': 'warning',
                        'DELETE': 'danger',
                        'PATCH': 'info'
                    };
                    return types[method] || 'info';
                };
                
                // API 调用
                const registerUser = async () => {
                    if (!userForm.userId || !userForm.userName) {
                        showMessage('参数错误', '用户ID和用户名不能为空', 'error');
                        return;
                    }
                    
                    loading.register = true;
                    try {
                        const response = await fetch(`/admin/users/${userForm.userId}/register?user_name=${encodeURIComponent(userForm.userName)}`, {
                            method: 'POST'
                        });
                        const data = await response.json();
                        
                        if (data.success) {
                            showMessage('注册成功', `用户 ${userForm.userName} 注册成功，端口: ${data.data?.port || 'N/A'}`, 'success');
                            addOperation('注册', userForm.userId, '成功', `容器已创建，端口: ${data.data?.port}`);
                            await listInstances();
                        } else {
                            showMessage('注册失败', data.detail || '未知错误', 'error');
                            addOperation('注册', userForm.userId, '失败', data.detail || '未知错误');
                        }
                    } catch (error) {
                        showMessage('网络错误', '请求失败，请检查网络连接', 'error');
                        addOperation('注册', userForm.userId, '失败', '网络请求失败');
                        console.error(error);
                    } finally {
                        loading.register = false;
                    }
                };
                
                const loginUser = async () => {
                    if (!userForm.userId) {
                        showMessage('参数错误', '用户ID不能为空', 'error');
                        return;
                    }
                    
                    loading.login = true;
                    try {
                        const response = await fetch(`/admin/users/${userForm.userId}/login`, {
                            method: 'POST'
                        });
                        const data = await response.json();
                        
                        if (data.success) {
                            showMessage('登录成功', `实例已启动，访问地址: ${data.url}`, 'success');
                            addOperation('登录', userForm.userId, '成功', `访问地址: ${data.url}`);
                            await listInstances();
                        } else {
                            showMessage('登录失败', data.detail || '未知错误', 'error');
                            addOperation('登录', userForm.userId, '失败', data.detail || '未知错误');
                        }
                    } catch (error) {
                        showMessage('网络错误', '请求失败，请检查网络连接', 'error');
                        addOperation('登录', userForm.userId, '失败', '网络请求失败');
                        console.error(error);
                    } finally {
                        loading.login = false;
                    }
                };
                
                const logoutUser = async () => {
                    if (!userForm.userId) {
                        showMessage('参数错误', '用户ID不能为空', 'error');
                        return;
                    }
                    
                    loading.logout = true;
                    try {
                        const response = await fetch(`/admin/users/${userForm.userId}/logout`, {
                            method: 'POST'
                        });
                        const data = await response.json();
                        
                        if (data.success) {
                            showMessage('登出成功', '用户实例已停止', 'success');
                            addOperation('登出', userForm.userId, '成功', '实例已停止');
                            await listInstances();
                        } else {
                            showMessage('登出失败', data.detail || '未知错误', 'error');
                            addOperation('登出', userForm.userId, '失败', data.detail || '未知错误');
                        }
                    } catch (error) {
                        showMessage('网络错误', '请求失败，请检查网络连接', 'error');
                        addOperation('登出', userForm.userId, '失败', '网络请求失败');
                        console.error(error);
                    } finally {
                        loading.logout = false;
                    }
                };
                
                const deleteUser = async () => {
                    if (!userForm.userId) {
                        showMessage('参数错误', '用户ID不能为空', 'error');
                        return;
                    }
                    
                    try {
                        await ElementPlus.ElMessageBox.confirm(
                            `确定要删除用户 "${userForm.userId}" 吗？此操作将删除用户的所有数据且不可恢复！`,
                            '删除确认',
                            {
                                confirmButtonText: '确定删除',
                                cancelButtonText: '取消',
                                type: 'warning'
                            }
                        );
                    } catch {
                        return;
                    }
                    
                    loading.delete = true;
                    try {
                        const response = await fetch(`/admin/users/${userForm.userId}`, {
                            method: 'DELETE'
                        });
                        const data = await response.json();
                        
                        if (data.success) {
                            showMessage('删除成功', '用户及其所有数据已删除', 'success');
                            addOperation('删除', userForm.userId, '成功', '用户及数据已删除');
                            userForm.userId = '';
                            userForm.userName = '';
                            await listInstances();
                        } else {
                            showMessage('删除失败', data.detail || '未知错误', 'error');
                            addOperation('删除', userForm.userId, '失败', data.detail || '未知错误');
                        }
                    } catch (error) {
                        showMessage('网络错误', '请求失败，请检查网络连接', 'error');
                        addOperation('删除', userForm.userId, '失败', '网络请求失败');
                        console.error(error);
                    } finally {
                        loading.delete = false;
                    }
                };
                
                const listInstances = async () => {
                    loading.list = true;
                    try {
                        const response = await fetch('/admin/instances');
                        const data = await response.json();
                        
                        if (data.success) {
                            instanceList.value = data.data || [];
                            updateStats();
                        } else {
                            showMessage('获取失败', data.detail || '获取实例列表失败', 'error');
                        }
                    } catch (error) {
                        console.error('获取实例列表失败:', error);
                    } finally {
                        loading.list = false;
                    }
                };
                
                const listBackups = async () => {
                    if (!backupForm.userId) {
                        showMessage('参数错误', '请输入用户ID', 'error');
                        return;
                    }
                    
                    loading.listBackups = true;
                    try {
                        const response = await fetch(`/admin/backups/${backupForm.userId}`);
                        const data = await response.json();
                        
                        if (data.success) {
                            backupList.value = data.data || [];
                            if (backupList.value.length === 0) {
                                showMessage('查询结果', '该用户暂无备份记录', 'info');
                            }
                        } else {
                            showMessage('获取失败', data.detail || '获取备份列表失败', 'error');
                        }
                    } catch (error) {
                        showMessage('网络错误', '获取备份列表失败', 'error');
                        console.error(error);
                    } finally {
                        loading.listBackups = false;
                    }
                };
                
                const createBackup = async () => {
                    if (!backupForm.userId) {
                        showMessage('参数错误', '请输入用户ID', 'error');
                        return;
                    }
                    
                    loading.createBackup = true;
                    try {
                        const response = await fetch(`/admin/backups/${backupForm.userId}/create`, {
                            method: 'POST'
                        });
                        const data = await response.json();
                        
                        if (data.success) {
                            showMessage('备份成功', `备份已创建: ${data.backup_path}`, 'success');
                            await listBackups();
                        } else {
                            showMessage('备份失败', data.detail || '未知错误', 'error');
                        }
                    } catch (error) {
                        showMessage('网络错误', '创建备份失败', 'error');
                        console.error(error);
                    } finally {
                        loading.createBackup = false;
                    }
                };
                
                const getMetrics = async () => {
                    if (!metricsForm.userId) {
                        showMessage('参数错误', '请输入用户ID', 'error');
                        return;
                    }
                    
                    loading.metrics = true;
                    try {
                        const response = await fetch(`/admin/metrics/${metricsForm.userId}`);
                        const data = await response.json();
                        
                        if (data.success) {
                            metricsData.value = data.data;
                        } else {
                            showMessage('获取失败', data.detail || '获取资源指标失败', 'error');
                        }
                    } catch (error) {
                        showMessage('网络错误', '获取资源指标失败', 'error');
                        console.error(error);
                    } finally {
                        loading.metrics = false;
                    }
                };
                
                // 生命周期
                onMounted(async () => {
                    // 隐藏加载提示
                    const loadingEl = document.getElementById('loading');
                    if (loadingEl) {
                        loadingEl.style.display = 'none';
                    }
                    loaded.value = true;
                    
                    // 加载实例列表
                    await listInstances();
                });
                
                return {
                    activeTab,
                    loaded,
                    message,
                    userForm,
                    backupForm,
                    metricsForm,
                    loading,
                    instanceList,
                    backupList,
                    metricsData,
                    recentOperations,
                    stats,
                    apiList,
                    handleMenuSelect,
                    showMessage,
                    registerUser,
                    loginUser,
                    logoutUser,
                    deleteUser,
                    listInstances,
                    listBackups,
                    createBackup,
                    getMetrics,
                    formatTime,
                    formatSize,
                    getProgressColor,
                    getMethodType
                };
            }
        });
        
        // 使用 Element Plus
        app.use(ElementPlus);
        
        // 挂载应用
        app.mount('#app');
    }
    
    // 页面加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initApp);
    } else {
        initApp();
    }
})();
