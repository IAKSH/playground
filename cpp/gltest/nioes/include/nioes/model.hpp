// Mesh,Model
// (VAO,VBO,EBO)

#include <glad/gles2.h>
#include <glm/matrix.hpp>
#include <tiny_gltf.h>
#include <unordered_map>
#include <string_view>
#include <functional>
#include <string>
#include <memory>
#include <vector>

#include <nioes/material.hpp>

namespace nioes {
    class Position {
    private:
        glm::vec3 xyz;
        
    public:
        Position(float x,float y,float z) noexcept;
        Position(glm::vec3 xyz) noexcept;
        Position(Position&) = default;
        Position() = default;
        ~Position() = default;

        float& x{xyz.x};
        float& y{xyz.y};
        float& z{xyz.z};

        float distance(const Position& pos) const noexcept;
        Position operator+ (const Position& pos) const noexcept;
        Position operator- (const Position& pos) const noexcept;
        bool operator== (const Position& pos) const noexcept;
        void operator= (const Position& pos) noexcept;
        void operator+= (const Position& pos) noexcept;
        void operator-=(const Position* pos) noexcept;
    };

    class Rotator {
    public:
        Rotator(float y,float p,float r) noexcept;
        Rotator(glm::vec3 ypr) noexcept;
        Rotator(Rotator&) = default;
        Rotator() = default;
        ~Rotator() = default;

        // 欧拉角
        glm::vec3 ypr;
        float& y{ypr.x};
        float& p{ypr.y};
        float& r{ypt.z};

        Position operator+ (const Rotator& pos) const noexcept;
        Position operator- (const Rotator& pos) const noexcept;
        bool operator== (const Rotator& pos) const noexcept;
        void operator= (const Rotator& pos) noexcept;
        void operator+= (const Rotator& pos) noexcept;
        void operator-=(const Rotator* pos) noexcept;
        
        glm::vec3 get_up_vec() noexcept;
        glm::vec3 get_right_vec() noexcept;
        glm::vec3 get_front_vec() noexcept;
    };

    // OpenGL (ES) Shader wrapper
    class Shader {
    private:
        GLuint id;
        void check_compile_status() noexcept(false);
		void compile(GLenum shader_type,std::string_view glsl) noexcept(false);

    public:
        Shader() noexcept;
        Shader(Shader&) = delete;
		~Shader() noexcept;
		GLuint get_id() const noexcept;
    };

    // OpenGL (ES) Shader Language Collection
    // 实现了基于预处理的GL_NIOES_include扩展
    class GLSLCollect {
    private:
        std::unordered_map<std::string, std::string> glsls;
		std::string impl_glsl_include(std::string_view glsl) noexcept(false);
		GLenum get_glsl_shader_type(std::string glsl) noexcept(false);
		bool check_shader_type_extension_enabled(std::string_view glsl) noexcept;
		void remove_cppoes_extensions(std::string& glsl) noexcept;

    public:
        GLSLCollect() = default;
		GLSLCollect(GLSLCollect&) = delete;
		~GLSLCollect() = default;
		void parse_from_file(std::string_view path) noexcept(false);
		void parse(std::string_view name, std::string_view glsl) noexcept(false);
		std::unique_ptr<Shader> compile(std::string_view name) noexcept(false);
    };

    template <typename T, typename U>
	constexpr bool is_same()
	{
		using Type = std::remove_cvref_t<T>;
		return std::is_same<Type, U>();
	}

	template <typename T, typename... Args>
	constexpr bool any_same()
	{
		if constexpr (sizeof...(Args) == 0)
			return false;
		else
			return (std::same_as<std::remove_reference_t<T>, Args> || ...) || any_same<Args...>();
	}

	template <typename T>
	concept Uniform = any_same<T, int, float, glm::vec2, glm::vec3, glm::vec4, glm::mat3, glm::mat4>();

    // OpenGL (ES) Shader Program wrapper
    class Program {
    private:
        GLuint id;
        void create_program(const Shader& vs, const Shader& fs) noexcept(false);
		void create_program(const Shader& vs, const Shader& gs, const Shader& fs) noexcept(false);
		GLint get_uniform_location(std::string_view uniform) noexcept(false);

    public:
        Program(const Shader& vs, const Shader& fs) noexcept(false);
		Program(const Shader& vs, const Shader& gs, const Shader& fs) noexcept(false);
        Program(Program&) = delete;
        ~Program() noexcept;

        GLuint get_program_id() const noexcept;
		void bind_uniform_block(std::string_view name, GLuint point) noexcept(false);

        template <Uniform T>
        void set_uniform(std::string_view uniform,const T& t) noexcept(false)
        {
            glUseProgram(program_id);

            GLint location{ get_uniform_location(uniform) };
			if constexpr (is_same<T, int>() || is_same<T, bool>())
				glUniform1i(location, t);
			else if constexpr (is_same<T, float>())
				glUniform1f(location, t);
			else if constexpr (is_same<T, glm::vec2>())
				glUniform2fv(location, 1, glm::value_ptr(t));
			else if constexpr (is_same<T, glm::vec3>())
				glUniform3fv(location, 1, glm::value_ptr(t));
			else if constexpr (is_same<T, glm::vec4>())
				glUniform4fv(location, 1, glm::value_ptr(t));
			else if constexpr (is_same<T, glm::mat3>())
				glUniformMatrix3fv(location, 1, false, glm::value_ptr(t));
			else if constexpr (is_same<T, glm::mat4>())
				glUniformMatrix4fv(location, 1, false, glm::value_ptr(t)); 
			else
				throw std::runtime_error("compile-time uniform type check failed");

            glUseProgram(0);
        }

        template <Uniform T>
        T get_uniform(std::string_view uniform) noexcept(false)
        {
            int size;
			unsigned int type;

			GLint location;
            if(location = get_uniform_location(uniform);!location)
                throw std::runtime_error(std::format("uniform {} not found from shader program {}",uniform,program_id));

			glGetActiveUniform(program_id, location, 0, nullptr, &size, &type, nullptr);

			unsigned int aimed_type;
			if constexpr (is_same<T, int>())
				aimed_type = GL_INT;
			else if constexpr (is_same<T, float>())
				aimed_type = GL_FLOAT;
			else if constexpr (is_same<T, glm::vec2>())
				aimed_type = GL_FLOAT_VEC2;
			else if constexpr (is_same<T, glm::vec3>())
				aimed_type = GL_FLOAT_VEC3;
			else if constexpr (is_same<T, glm::vec4>())
				aimed_type = GL_FLOAT_VEC4;
			else if constexpr (is_same<T, glm::mat3>())
				aimed_type = GL_FLOAT_MAT3;
			else if constexpr (is_same<T, glm::mat4>())
				aimed_type = GL_FLOAT_MAT4;
			else
				throw std::runtime_error("compile-time uniform type check failed");

			if (size != 1)
				throw std::runtime_error("uniform name conflict");
			if (type != aimed_type)
				throw("unsupported uniform type");

			T ret_uniform;
			if constexpr (is_same<T, int>())
				glGetUniformiv(program_id, location, &ret_uniform);
			else if constexpr (is_same<T, float>())
				glGetUniformfv(program_id, location, &ret_uniform);
			else
				glGetUniformfv(program_id, location, glm::value_ptr(ret_uniform));

			return ret_uniform;
        }
    };

    // OpenGL (ES) UBO/SSBO wrapper
    class GPUBuffer {
    private:
        GLuint id;
        GLenum target;
        GLenum usage;

        bool verifiy_target(GLenum target) const noexcept;
        bool verifiy_usage(GLenum usage) const noexcept;

        void gen_buffer_id() noexcept;
        void pre_allocate_mem(GLsizeiptr size) noexcept;
        void delete_buffer() noexcept;

    public:
        GPUBuffer(GLenum target, GLenum usage, GLsizeiptr size) noexcept(false);
        GPUBuffer(GPUBuffer& src, GLsizeiptr new_size = 0) noexcept;
        ~GPUBuffer() noexcept;

        GLuint get_buffer_id() const noexcept;
        GLsizeiptr get_buffer_size() const noexcept;
        GLenum get_buffer_target() const noexcept;
        GLenum get_buffer_usage() const noexcept;
        void set_buffer_target(GLenum new_target) noexcept(false);

        // this will not check if out of range
        void write_buffer_data(const void* data, GLintptr offset, GLsizeiptr size) noexcept;
        // this will not check if out of range
        void dma_do(std::function<void(void* data)> callback, GLintptr offset, GLsizeiptr length, GLbitfield access) noexcept;
        void dma_do(std::function<void(void* data)> callback) noexcept;

        template <typename T>
        requires requires(T t)
        {
            {t.size()} -> std::same_as<std::size_t>;
            {t.data()} -> std::convertible_to<void*>;
        }
        void write_buffer_data(const T& t, std::size_t size = 0) noexcept
        {
            if (size)
                write_buffer_data(t.data(), 0, size);
            else
                write_buffer_data(t.data(), 0, t.size() * sizeof(float));
        }
    };

    // OpenGL (ES) VBO wrapper
    class VertexBuffer : public GPUBuffer {
    public:
        VertexBuffer(GLenum usage,GLsizeiptr size) noexcept
            : GPUBuffer(GL_VERTEX_ARRAY,usage,size) {}
    };

    // OpenGL (ES) EBO wrapper
    class ElementBuffer : public GPUBuffer {
    public:
        ElementBuffer(GLenum usage,GLsizeiptr size) noexcept
            : GPUBuffer(GL_ELEMENT_ARRAY_BUFFER,usage,size) {}
    };

    // OpenGL (ES) UBO wrapper
    class UniformBuffer : public GPUBuffer {
    public:
        UniformBuffer(GLenum usage,GLsizeiptr size) noexcept
            : GPUBuffer(GL_UNIFORM_BUFFER,usage,size) {}
    };

    // OpenGL (ES) SSBO wrapper
    class ShaderStoreBuffer : public GPUBuffer {
    public:
        ShaderStoreBuffer(GLenum usage,GLsizeiptr size) noexcept
            : GPUBuffer(GL_SHADER_STORAGE_BUFFER,usage,size) {}
    };

    // OpenGL (ES) VAO wrapper
    class VertexArray {
    public:
        private:
        GLuint vao_id;
        bool has_ebo;

        void create_vao() noexcept;
        void delete_vao() noexcept;

    public:
        VertexArray() noexcept;
        VertexArray(VertexArray&) = delete;
        ~VertexArray() noexcept;

        GLuint get_vao_id() const noexcept;
        void bind_ebo(GLuint ebo_id) noexcept;
        void draw(GLuint program_id, GLenum primitive, GLint first, GLsizei vertex_count, GLsizei instance = 0) noexcept;
        void add_attrib(GLenum buffer_target, uint32_t buffer_id,
            uint32_t index, uint32_t len, uint32_t vertex_len, uint32_t offset, bool normalized = false) noexcept(false);
        void bind_ebo(const ElementBuffer& ebo) noexcept(false);
        void add_attrib(const VertexBuffer& vbo, uint32_t index, uint32_t len, uint32_t vertex_len, uint32_t offset, bool normalized = false) noexcept(false);
        void draw(const Program& program, GLenum primitive, GLint first, GLsizei vertex_count, GLsizei instance = 0) noexcept;
    };

    struct PackedTexture
    {
        std::unique_ptr<Texture> texture;
        std::string type;
        std::string path;
        PackedTexture(std::string_view file_name, std::string_view directory, std::string_view type) noexcept(false);
    };

    inline static constexpr uint32_t MAX_BONE_INFLUENCE{ 4 };
    struct PackedVertex
    {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec3 tangent;
        glm::vec3 bitangent;
        int bone_ids[MAX_BONE_INFLUENCE];
        float weights[MAX_BONE_INFLUENCE];
    };

    class Mesh {
    private:
        std::vector<PackedTexture*> textures;
        std::unique_ptr<VertexBuffer> vbo;
        std::unique_ptr<ElementBuffer> ebo;
        std::size_t indices_len;
        VertexArray vao;

        GLenum primitive_type;

        void setup_vao(std::vector<PackedTexture>& vertices,
            std::vector<unsigned int>& indices) noexcept;

    public:
        Mesh(std::vector<PackedTexture*>& textures,
            std::vector<PackedVertex>& vertices,
            std::vector<unsigned int>& indices,
            GLenum primivite_type = GL_TRIANGLES) noexcept;

        Mesh(Mesh&) = delete;
        ~Mesh() = default;

        VertexArray& get_vao() noexcept;
        VertexBuffer& get_vbo() noexcept;
        ElementBuffer& get_ebo() noexcept;
        std::vector<PackedTexture*>& get_textures() noexcept;
        void set_indices_size(std::size_t indices_len) noexcept;
        void draw_mesh(Program& program, GLenum primitive = GL_NONE, uint32_t instance = 0) noexcept;
    };

    // 模型资源，应当全局唯一
    class Model {
    private:
        std::vector<Mesh> meshes;
        void foo() {
            tinygltf::Texture tex;
        }
    };

    // 骨骼动画资源，应当全局唯一
    // 动画进行状态储存在Object，而不是这里
    class Animation {

    };

    // TODO: 可能需要为模型与动画提供工厂类，或者是工厂函数
    // 以提供Image（Texture）以及其他类似的，可以复用的资源的缓存

    // 模型，材质和动画的载体
    class Object {
    private:
        // TODO: 模型，材质和动画

    public:
        Object() = default;
        Object(Object&) = default;
        ~Object() = default;

        Position pos;
        Rotator rotate;
    };

    class Camera : public Object {
    
    };
}