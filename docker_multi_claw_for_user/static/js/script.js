// 初始化Vue应用
const { createApp } = Vue;
const app = createApp({
    data() {
        return {
            activeTab: '1',
            // 用户表单
            userForm: {
                userId: '',
                userName: ''
            },
            // 备份表单
            backupForm: {
                userId: ''
            },
            // 监控表单
            metricsForm: {
                userId: '',
                metricsData: null
            },
            // 消息提示
            message: null,
            // 实例列表
            instanceList: [],
            // 备份列表
            backupList: [],
            // 监控数据
            metricsData: null
        };
    },
    methods: {
        // 显示消息提示
        showMessage(title, content, type) {
            this.message = {
                title: title,
                content: content,
                type: type
            };
        },
        // 注册用户
        async registerUser() {
            if (!this.userForm.userId || !this.userForm.userName) {
                this.showMessage('错误', '用户ID和用户名不能为空', 'error');
                return;
            }
            try {
                const response = await fetch(`/admin/users/${this.userForm.userId}/register?user_name=${this.userForm.userName}`, {
                    method: 'POST'
                });
                const data = await response.json();
                if (data.success) {
                    this.showMessage('成功', '用户注册成功', 'success');
                } else {
                    this.showMessage('错误', data.detail || '注册失败', 'error');
                }
            } catch (error) {
                this.showMessage('错误', '网络请求失败', 'error');
                console.error(error);
            }
        },
        // 用户登录
        async loginUser() {
            if (!this.userForm.userId) {
                this.showMessage('错误', '用户ID不能为空', 'error');
                return;
            }
            try {
                const response = await fetch(`/admin/users/${this.userForm.userId}/login`, {
                    method: 'POST'
                });
                const data = await response.json();
                if (data.success) {
                    this.showMessage('成功', `登录成功，访问地址：${data.url}`, 'success');
                } else {
                    this.showMessage('错误', data.detail || '登录失败', 'error');
                }
            } catch (error) {
                this.showMessage('错误', '网络请求失败', 'error');
                console.error(error);
            }
        },
        // 用户登出
        async logoutUser() {
            if (!this.userForm.userId) {
                this.showMessage('错误', '用户ID不能为空', 'error');
                return;
            }
            try {
                const response = await fetch(`/admin/users/${this.userForm.userId}/logout`, {
                    method: 'POST'
                });
                const data = await response.json();
                if (data.success) {
                    this.showMessage('成功', '用户登出成功', 'success');
                } else {
                    this.showMessage('错误', data.detail || '登出失败', 'error');
                }
            } catch (error) {
                this.showMessage('错误', '网络请求失败', 'error');
                console.error(error);
            }
        },
        // 删除用户
        async deleteUser() {
            if (!this.userForm.userId) {
                this.showMessage('错误', '用户ID不能为空', 'error');
                return;
            }
            try {
                const response = await fetch(`/admin/users/${this.userForm.userId}`, {
                    method: 'DELETE'
                });
                const data = await response.json();
                if (data.success) {
                    this.showMessage('成功', '用户删除成功', 'success');
                } else {
                    this.showMessage('错误', data.detail || '删除失败', 'error');
                }
            } catch (error) {
                this.showMessage('错误', '网络请求失败', 'error');
                console.error(error);
            }
        },
        // 列出所有实例
        async listInstances() {
            try {
                const response = await fetch('/admin/instances');
                const data = await response.json();
                if (data.success) {
                    this.instanceList = data.data;
                    this.showMessage('成功', '实例列表刷新成功', 'success');
                } else {
                    this.showMessage('错误', data.detail || '获取实例列表失败', 'error');
                }
            } catch (error) {
                this.showMessage('错误', '网络请求失败', 'error');
                console.error(error);
            }
        },
        // 列出用户备份
        async listBackups() {
            if (!this.backupForm.userId) {
                this.showMessage('错误', '用户ID不能为空', 'error');
                return;
            }
            try {
                const response = await fetch(`/admin/backups/${this.backupForm.userId}`);
                const data = await response.json();
                if (data.success) {
                    this.backupList = data.data;
                    this.showMessage('成功', '备份列表获取成功', 'success');
                } else {
                    this.showMessage('错误', data.detail || '获取备份列表失败', 'error');
                }
            } catch (error) {
                this.showMessage('错误', '网络请求失败', 'error');
                console.error(error);
            }
        },
        // 创建用户备份
        async createBackup() {
            if (!this.backupForm.userId) {
                this.showMessage('错误', '用户ID不能为空', 'error');
                return;
            }
            try {
                const response = await fetch(`/admin/backups/${this.backupForm.userId}/create`, {
                    method: 'POST'
                });
                const data = await response.json();
                if (data.success) {
                    this.showMessage('成功', `备份创建成功，路径：${data.backup_path}`, 'success');
                    // 刷新备份列表
                    this.listBackups();
                } else {
                    this.showMessage('错误', data.detail || '创建备份失败', 'error');
                }
            } catch (error) {
                this.showMessage('错误', '网络请求失败', 'error');
                console.error(error);
            }
        },
        // 获取用户资源指标
        async getMetrics() {
            if (!this.metricsForm.userId) {
                this.showMessage('错误', '用户ID不能为空', 'error');
                return;
            }
            try {
                const response = await fetch(`/admin/metrics/${this.metricsForm.userId}`);
                const data = await response.json();
                if (data.success) {
                    this.metricsData = data.data;
                    this.showMessage('成功', '资源指标获取成功', 'success');
                } else {
                    this.showMessage('错误', data.detail || '获取资源指标失败', 'error');
                }
            } catch (error) {
                this.showMessage('错误', '网络请求失败', 'error');
                console.error(error);
            }
        }
    },
    mounted() {
        // 初始化Element Plus图标
        for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
            app.component(key, component);
        }
    }
});

// 注册Element Plus
app.use(ElementPlus);
app.mount('#app');