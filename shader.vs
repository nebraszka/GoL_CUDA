#version 330 core
void main() {
    float x = (gl_VertexID & 1) * 2.0f - 1.0f;
    float y = ((gl_VertexID & 2) >> 1) * 2.0f - 1.0f;
    gl_Position = vec4(x, y, 0.0f, 1.0f);
}