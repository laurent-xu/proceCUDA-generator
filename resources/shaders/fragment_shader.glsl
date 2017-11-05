#version 450 core

out vec4 color;

uniform vec3 objectColor;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewerPos;

in vec3 fwd_normal;
in vec3 fragPos;

void main() {
    float ambient_strenght = 0.1f;
    float specular_strenght = 0.5f;
    vec3 ambientLight = lightColor * ambient_strenght;
    vec3 nNormal = normalize(fwd_normal);
    vec3 nFragPos = normalize(lightPos - fragPos);
    vec3 viewDir = normalize(viewerPos - fragPos);
    vec3 reflectDir = reflect(-nFragPos, nNormal);
    float specConst = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specLight = specular_strenght * specConst * lightColor;
    float diffLight = max(dot(nNormal, nFragPos), 0.0f);
    vec3 light = diffLight + ambientLight + specLight;
    color = vec4(light * objectColor, 1.0f);
}
