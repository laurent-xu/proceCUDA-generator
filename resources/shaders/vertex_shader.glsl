#version 450 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fwd_normal;
out vec3 fragPos;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0f);
    fwd_normal = mat3(transpose(inverse(model))) * normal;
    fragPos = vec3(model * vec4(position, 1.0f));
}
