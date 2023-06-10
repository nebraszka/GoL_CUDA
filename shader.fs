#version 330 core
uniform sampler2D tex;
out vec4 FragColor;

void main() {
   FragColor = texture(tex, gl_FragCoord.xy / vec2(textureSize(tex, 0)));
}