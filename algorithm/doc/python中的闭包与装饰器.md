### Python中的闭包与装饰器详解

#### 一、闭包（Closure）
**定义**：闭包是**携带外部状态的函数**，当一个内部函数引用了外部作用域的变量时，即使外部函数执行完毕，这些变量也不会被销毁。

**核心要素**：
1. 嵌套函数（内部函数定义在外部函数内）
2. 内部函数引用外部函数的变量
3. 外部函数返回内部函数

**示例**：
```python
def outer_func(x):
    # 外部变量
    message = "Value: "
    
    def inner_func(y):
        # 引用外部变量（闭包形成的关键）
        return f"{message}{x + y}"
    
    return inner_func  # 返回内部函数（非调用）

closure = outer_func(10)
print(closure(5))  # 输出: "Value: 15"
```

**特点**：
- 保持状态：`closure` 记住了 `x=10` 和 `message`
- 延迟执行：返回函数对象而非立即执行
- 数据封装：外部变量对全局空间不可见

---

#### 二、装饰器（Decorator）
**定义**：装饰器是**修改或增强函数行为的工具**，本质上是接受函数作为参数并返回新函数的可调用对象。

**基本结构**：
```python
def decorator(func):
    def wrapper(*args, **kwargs):
        # 1. 前置处理
        result = func(*args, **kwargs)  # 调用原函数
        # 2. 后置处理
        return result
    return wrapper

@decorator
def target_function():
    ...
```

**等效调用**：
```python
target_function = decorator(target_function)
```

**带参数装饰器**：
```python
def repeat(n):
    def actual_decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return actual_decorator

@repeat(3)
def say_hello():
    print("Hello!")

say_hello()  # 输出3次"Hello!"
```

---

### 三、闭包与装饰器的联系
1. **实现基础**：
   - 装饰器本质上就是闭包的应用
   - 装饰器的 `wrapper` 函数引用了外部作用域的 `func` 参数

2. **状态保持**：
   ```python
   def counter_decorator(func):
       count = 0  # 闭包保存状态
       def wrapper(*args, **kwargs):
           nonlocal count
           count += 1
           print(f"Call count: {count}")
           return func(*args, **kwargs)
       return wrapper
   ```

3. **实现关系**：
   ```mermaid
   graph LR
   A[闭包] --> B[携带状态]
   B --> C[装饰器基础]
   C --> D[函数增强]
   ```

---

### 四、适用场景
#### 闭包典型场景
1. **状态保持**：
   ```python
   def make_adder(x):
       def adder(y):
           return x + y
       return adder
   
   add5 = make_adder(5)
   print(add5(3))  # 输出8
   ```

2. **回调函数**：
   ```python
   def event_handler(button):
       def on_click():
           print(f"{button} clicked!")
       return on_click
   
   btn1_handler = event_handler("Button1")
   ```

3. **函数工厂**：
   ```python
   def power_factory(exponent):
       def power(base):
           return base ** exponent
       return power
   
   square = power_factory(2)
   cube = power_factory(3)
   ```

#### 装饰器典型场景
1. **功能增强**：
   ```python
   def log_execution(func):
       def wrapper(*args, **kwargs):
           print(f"Executing {func.__name__}")
           result = func(*args, **kwargs)
           print(f"Finished {func.__name__}")
           return result
       return wrapper
   ```

2. **访问控制**：
   ```python
   def login_required(func):
       def wrapper(user, *args, **kwargs):
           if not user.is_authenticated:
               raise PermissionError("Login required")
           return func(user, *args, **kwargs)
       return wrapper
   ```

3. **性能优化**：
   ```python
   import time
   def timer(func):
       def wrapper(*args, **kwargs):
           start = time.perf_counter()
           result = func(*args, **kwargs)
           duration = time.perf_counter() - start
           print(f"{func.__name__} took {duration:.4f} seconds")
           return result
       return wrapper
   ```

4. **注册机制**：
   ```python
   PLUGINS = {}
   def register(func):
       PLUGINS[func.__name__] = func
       return func
   
   @register
   def plugin1(): ...
   ```

---

### 五、高级应用技巧
1. **多层装饰器**：
   ```python
   @decorator1
   @decorator2
   def func(): ...
   # 等效于: func = decorator1(decorator2(func))
   ```

2. **保留元数据**：
   ```python
   import functools
   def decorator(func):
       @functools.wraps(func)  # 保留原函数信息
       def wrapper(*args, **kwargs):
           ...
       return wrapper
   ```

3. **类装饰器**：
   ```python
   class CountCalls:
       def __init__(self, func):
           self.func = func
           self.count = 0
       
       def __call__(self, *args, **kwargs):
           self.count += 1
           return self.func(*args, **kwargs)
   
   @CountCalls
   def example(): ...
   ```

---

### 六、选择指南
| **场景**                  | **闭包** | **装饰器** |
|---------------------------|----------|------------|
| 需要创建有状态函数        | ✓        |            |
| 需要修改函数行为          |          | ✓          |
| 需要函数注册机制          |          | ✓          |
| 需要延迟执行              | ✓        |            |
| 需要代码复用/DRY原则      |          | ✓          |
| 需要临时函数工厂          | ✓        |            |

### 总结
- **闭包**：用于**创建携带状态的函数**，解决状态保持问题
- **装饰器**：基于闭包实现**函数行为修改**，解决横切关注点问题
- **联系**：装饰器是闭包的典型应用，二者共同实现函数的动态增强
- **最佳实践**：
  - 简单状态保持 → 闭包
  - 函数行为扩展 → 装饰器
  - 复杂场景 → 类装饰器（结合OOP）

通过合理使用闭包和装饰器，可以显著提升代码的可重用性、可维护性和可扩展性，是Python进阶必备的核心技术。