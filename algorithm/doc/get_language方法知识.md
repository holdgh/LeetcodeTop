`django.utils.translation.get_language()` 是 Django 国际化（i18n）和本地化（l10n）框架中的一个核心函数，**用于获取当前请求的激活语言代码**（例如 `'en-us'`、`'zh-hans'`）。它的主要作用是为开发者提供一种便捷的方式，在代码中动态判断或处理与当前语言相关的逻辑。

---

### 核心作用
1. **获取当前语言环境**  
   返回 Django 根据请求自动检测或用户显式设置的**当前语言代码**（遵循标准格式，如 `语言-地区`）。
   ```python
   from django.utils.translation import get_language

   current_lang = get_language()
   print(current_lang)  # 例如 'en-us' 或 'zh-hans'
   ```

2. **支持多语言逻辑分支**  
   在代码中根据语言动态调整行为，例如：
   ```python
   if get_language() == 'zh-hans':
       # 针对简体中文的逻辑
   else:
       # 其他语言的默认逻辑
   ```

3. **与翻译系统集成**  
   结合 `gettext()` 等翻译函数，实现动态内容切换：
   ```python
   from django.utils.translation import gettext as _

   greeting = _("Hello")  # 根据当前语言返回翻译后的文本
   ```

---

### 如何确定当前语言？
Django 通过 `LocaleMiddleware` 中间件确定当前语言，优先级顺序为：
1. **URL 前缀**（如 `/zh-hans/about/`）
2. **用户会话（session）中的 `django_language` 键**
3. **Cookie 中的 `django_language`**
4. **HTTP 请求头 `Accept-Language`**
5. **全局默认语言 `LANGUAGE_CODE`（`settings.py` 中设置）**

若未启用 `LocaleMiddleware`，`get_language()` 默认返回 `LANGUAGE_CODE`。

---

### 常见使用场景
1. **动态内容选择**  
   根据语言查询不同数据库内容：
   ```python
   def get_article(request):
       lang = get_language()
       article = Article.objects.filter(language=lang).first()
       return render(request, 'article.html', {'article': article})
   ```

2. **API 响应本地化**  
   在 REST API 中返回对应语言的数据：
   ```python
   class ProductView(APIView):
       def get(self, request):
           lang = get_language()
           products = Product.objects.filter(lang=lang)
           return Response(ProductSerializer(products).data)
   ```

3. **日志或分析**  
   记录用户使用的语言：
   ```python
   def log_user_action(user, action):
       lang = get_language()
       UserLog.objects.create(user=user, action=action, language=lang)
   ```

---

### 相关函数
- **`get_language_info(lang_code)`**:  
  获取语言的详细信息（如名称、本地名称、方向等）。
  ```python
  from django.utils.translation import get_language_info

  info = get_language_info('zh-hans')
  print(info['name'])         # 'Chinese'
  print(info['name_local'])   # '简体中文'
  ```

- **`activate(language)`**:  
  手动设置当前线程的语言（通常用于临时覆盖）：
  ```python
  from django.utils.translation import activate

  activate('fr')  # 后续翻译将使用法语
  ```

- **模板中的等效操作**：  
  在模板中通过 `{% get_current_language as LANGUAGE_CODE %}` 获取语言代码。

---

### 注意事项
1. **中间件依赖**  
   确保 `django.middleware.locale.LocaleMiddleware` 已添加到 `settings.MIDDLEWARE`，且顺序在 `SessionMiddleware` 和 `CookieMiddleware` 之后。

2. **线程安全**  
   `get_language()` 返回的是当前线程的语言，因此在异步或多线程环境中需谨慎处理。

3. **缓存优化**  
   频繁调用 `get_language()` 可能影响性能，建议在需要时缓存结果：
   ```python
   lang = get_language()
   ```

---

### 示例：多语言欢迎页面
```python
from django.http import HttpResponse
from django.utils.translation import get_language

def welcome(request):
    lang = get_language()
    messages = {
        'en': 'Welcome!',
        'zh-hans': '欢迎！',
        'es': '¡Bienvenido!'
    }
    return HttpResponse(messages.get(lang, messages['en']))
```

通过 `get_language()`，Django 应用能灵活适配不同语言场景，是构建国际化 Web 应用的关键工具。