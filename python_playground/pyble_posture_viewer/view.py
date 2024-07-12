from bluepy.btle import Peripheral, UUID
import struct
import sys
import glfw
import OpenGL.GL as gl
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
import glm
import numpy as np

p = Peripheral("30:30:F9:72:00:9A", "public")

def bytes_to_floats(b):
    floats = []
    for i in range(0, len(b), 4):
        floats.append(struct.unpack('f', b[i:i+4])[0])
    return floats

def get_data(uuid):
    try:
        ch = p.getCharacteristics(uuid=UUID(uuid))[0]
        if (ch.supportsRead()):
            return bytes_to_floats(ch.read())
        else:
            print(f"The characteristic {uuid} does not support read operation")
    except Exception as e:
        print(f"Error reading the characteristic {uuid}: {str(e)}")

def on_key(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window,1)

# Initialize the library
if not glfw.init():
    sys.exit()

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(640, 480, "Hello World", None, None)
if not window:
    glfw.terminate()
    sys.exit()

# Make the window's context current
glfw.make_context_current(window)

# Install a key handler
glfw.set_key_callback(window, on_key)

vertex_shader = """
#version 330
layout(location = 0) in vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

fragment_shader = """
#version 330
out vec4 fragColor;
void main()
{
    fragColor = vec4(1.0, 0.5, 0.0, 1.0);
}
"""

# 编译着色器
vs = shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER)
fs = shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)

# 创建着色器程序
shader = shaders.compileProgram(vs, fs)

# 创建一个简单的立方体
vertices = np.array([
    -0.5, -0.5, -0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5, -0.5, -0.5, -0.5, -0.5,
    -0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,
    -0.5, -0.5, -0.5,  0.5, -0.5, -0.5,  0.5, -0.5,  0.5,  0.5, -0.5,  0.5, -0.5, -0.5,  0.5, -0.5, -0.5, -0.5,
    -0.5,  0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5, -0.5,  0.5,  0.5, -0.5,  0.5, -0.5,
    -0.5, -0.5, -0.5, -0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5, -0.5, -0.5, -0.5,
     0.5, -0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5, -0.5,  0.5,  0.5, -0.5, -0.5
], dtype=np.float32)

# 创建VBO和VAO
vbo = vbo.VBO(np.array(vertices, dtype=np.float32))
vao = gl.glGenVertexArrays(1)
gl.glBindVertexArray(vao)
vbo.bind()
gl.glEnableVertexAttribArray(0)
gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 0, None)
vbo.unbind()
gl.glBindVertexArray(0)
gl.glEnable(gl.GL_DEPTH_TEST)
gl.glDepthFunc(gl.GL_LESS)

gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

while not glfw.window_should_close(window):
    # 渲染立方体
    gl.glUseProgram(shader)
    gl.glBindVertexArray(vao)

    # 更新模型矩阵
    euler = get_data("0000ff02-0000-1000-8000-00805f9b34fb")
    print(euler)
    model = glm.mat4(1.0)
    model = glm.rotate(model, glm.radians(euler[0]), glm.vec3(1.0, 0.0, 0.0))
    model = glm.rotate(model, glm.radians(euler[1]), glm.vec3(0.0, 1.0, 0.0))
    model = glm.rotate(model, glm.radians(euler[2]), glm.vec3(0.0, 0.0, 1.0))

    view = glm.lookAt(glm.vec3(0.0, 0.0, 3.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
    projection = glm.perspective(glm.radians(45.0), 1280.0 / 720.0, 0.1, 100.0)
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "model"), 1, gl.GL_FALSE, glm.value_ptr(model))
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "view"), 1, gl.GL_FALSE, glm.value_ptr(view))
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "projection"), 1, gl.GL_FALSE, glm.value_ptr(projection))
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)
    gl.glBindVertexArray(0)
    gl.glUseProgram(0)

    glfw.swap_buffers(window)
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

p.disconnect()
glfw.terminate()

