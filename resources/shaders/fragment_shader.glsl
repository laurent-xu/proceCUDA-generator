#version 300 es

out lowp vec4 color;

uniform lowp vec3 pointLightPos;
uniform lowp vec3 pointLightColor;
uniform lowp vec3 objectColor;
uniform lowp vec3 lightColor1;
uniform lowp vec3 lightDir1;
uniform lowp vec3 viewerPos;

in lowp vec3 fwd_normal;
in lowp vec3 fragPos;

void main() {
    lowp float ambient_strenght = 0.4f;
    lowp float specular_strenght = 0.5f;


    lowp vec3 nNormal = normalize(fwd_normal);

    lowp vec3 ambientLight = lightColor1 * ambient_strenght;
    lowp vec3 viewDir = normalize(viewerPos - fragPos);
    lowp vec3 nLightDir1 = normalize(lightDir1);
    lowp float diffLight1 = max(dot(nNormal, nLightDir1), 0.0f);


    lowp float distance    = length(viewerPos - fragPos);
    lowp float attenuation = 1.0f / (1.0f + 0.007f * distance + 0.0002f * (distance * distance));
    lowp vec3 dirPointLight = normalize(pointLightPos - fragPos);
    lowp float pointDiffLight = max(dot(nNormal, dirPointLight), 0.0f);

    lowp vec3 light = pointDiffLight * attenuation * lightColor1 + /* diffLight1 * 0.7 * lightColor1 +*/ ambientLight;
    color = vec4(light * objectColor, 1.0f);
}
