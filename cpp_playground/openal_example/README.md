似乎vcpkg的mpg123:x64-mingw-dynamic有些问题，mpg123_open()会找不到实现

换到x64-mingw-static就好了，但是问题是全部静态链接会很花时间，mingw-gcc的调试信息还很大，花的时间更多了。

x64-windows就没有任何问题，甚至能自动配置那些库用dynamic，哪些库用static

不知道vcpkg.json有没有办法为每一个依赖指定dynamic还是static，如果有的话就太好了
