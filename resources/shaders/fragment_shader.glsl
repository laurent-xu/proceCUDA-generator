#version 300 es

out lowp vec4 color;

uniform lowp vec3 objectColor;
uniform lowp vec3 lightColor;
uniform lowp vec3 lightPos;
uniform lowp vec3 viewerPos;

in lowp vec3 fwd_normal;
in lowp vec3 fragPos;

void main() {
    lowp float ambient_strenght = 0.1f;
    lowp float specular_strenght = 0.5f;
    lowp vec3 ambientLight = lightColor * ambient_strenght;
    lowp vec3 nNormal = normalize(fwd_normal);
    lowp vec3 nFragPos = normalize(lightPos - fragPos);
    lowp vec3 viewDir = normalize(viewerPos - fragPos);
    lowp vec3 reflectDir = reflect(-nFragPos, nNormal);
    lowp float specConst = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    lowp vec3 specLight = specular_strenght * specConst * lightColor;
    lowp float diffLight = max(dot(nNormal, nFragPos), 0.0f);
    lowp vec3 light = diffLight + ambientLight + specLight;
    // color = vec4(light * objectColor, 1.0f);
    color = vec4(1.0, 0.0, 0.0, 1.0f);
}
