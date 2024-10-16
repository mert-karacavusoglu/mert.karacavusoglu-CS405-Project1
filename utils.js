function multiplyMatrices(matrixA, matrixB) {
    var result = [];

    for (var i = 0; i < 4; i++) {
        result[i] = [];
        for (var j = 0; j < 4; j++) {
            var sum = 0;
            for (var k = 0; k < 4; k++) {
                sum += matrixA[i * 4 + k] * matrixB[k * 4 + j];
            }
            result[i][j] = sum;
        }
    }

    // Flatten the result array
    return result.reduce((a, b) => a.concat(b), []);
}
function createIdentityMatrix() {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
}
function createScaleMatrix(scale_x, scale_y, scale_z) {
    return new Float32Array([
        scale_x, 0, 0, 0,
        0, scale_y, 0, 0,
        0, 0, scale_z, 0,
        0, 0, 0, 1
    ]);
}

function createTranslationMatrix(x_amount, y_amount, z_amount) {
    return new Float32Array([
        1, 0, 0, x_amount,
        0, 1, 0, y_amount,
        0, 0, 1, z_amount,
        0, 0, 0, 1
    ]);
}

function createRotationMatrix_Z(radian) {
    return new Float32Array([
        Math.cos(radian), -Math.sin(radian), 0, 0,
        Math.sin(radian), Math.cos(radian), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_X(radian) {
    return new Float32Array([
        1, 0, 0, 0,
        0, Math.cos(radian), -Math.sin(radian), 0,
        0, Math.sin(radian), Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_Y(radian) {
    return new Float32Array([
        Math.cos(radian), 0, Math.sin(radian), 0,
        0, 1, 0, 0,
        -Math.sin(radian), 0, Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function getTransposeMatrix(matrix) {
    return new Float32Array([
        matrix[0], matrix[4], matrix[8], matrix[12],
        matrix[1], matrix[5], matrix[9], matrix[13],
        matrix[2], matrix[6], matrix[10], matrix[14],
        matrix[3], matrix[7], matrix[11], matrix[15]
    ]);
}

const vertexShaderSource = `
attribute vec3 position;
attribute vec3 normal; // Normal vector for lighting

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 normalMatrix;

uniform vec3 lightDirection;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vNormal = vec3(normalMatrix * vec4(normal, 0.0));
    vLightDirection = lightDirection;

    gl_Position = vec4(position, 1.0) * projectionMatrix * modelViewMatrix; 
}

`

const fragmentShaderSource = `
precision mediump float;

uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float shininess;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(vLightDirection);
    
    // Ambient component
    vec3 ambient = ambientColor;

    // Diffuse component
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor;

    // Specular component (view-dependent)
    vec3 viewDir = vec3(0.0, 0.0, 1.0); // Assuming the view direction is along the z-axis
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * specularColor;

    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
}

`

/**
 * @WARNING DO NOT CHANGE ANYTHING ABOVE THIS LINE
 */ 



/**
 * 
 * @TASK1 Calculate the model view matrix by using the chatGPT
 */

function getChatGPTModelViewMatrix() {
    const transformationMatrix = new Float32Array([
        0.1767767, -0.3061862,  0.3535534,  0.3,
        0.3838835,  0.4330127, -0.1767767, -0.25,
        -0.25,      0.1767767,  0.4330127,  0,
        0,          0,          0,          1
    ]);
    return getTransposeMatrix(transformationMatrix);
}


/**
 * 
 * @TASK2 Calculate the model view matrix by using the given 
 * transformation methods and required transformation parameters
 * stated in transformation-prompt.txt
 */
function getModelViewMatrix() {
    // calculate the model view matrix by using the transformation
    // methods and return the modelView matrix in this method

    //We will use the functions provided to us in this file to compute the transformation matrix
    //First we will need to calculate the individual tranformation matrices, and then we will multiply them together to get the final
    //transformation matrix

    //Translation matrix
    //We call the createTranslationMatrix() function with the parameters given in the tranformation-prompt.txt file
    translationMatrix = getTransposeMatrix(createTranslationMatrix(0.3, -0.25, 0.0));  //X = 0.3, //Y = -0.25 //Z = 0.0

    //Scaling matrix
    //We call the createScaleMatrix() function with the parameters given in the transformation-promt.txt filse
    scaleMatrix = createScaleMatrix(0.5,0.5,1); //X = 0.5, //Y = 0.5, //Z = 1.0 (So Z won't change)

    //Rotation matrices
    //We Call the createRotation matrix functions for each direction, 
    
    rotMatrix_X = createRotationMatrix_X(Math.PI/6.0); //Pi/6 radians = 30 degrees
    rotMatrix_Y = createRotationMatrix_Y(Math.PI/4.0); //Pi/4 radians = 45 degrees
    rotMatrix_Z = createRotationMatrix_Z(Math.PI/3.0); //Pi/3 radians = 60 degrees
    
    transformationMatrix = createIdentityMatrix();
    transformationMatrix = multiplyMatrices(transformationMatrix, scaleMatrix); 
    transformationMatrix = multiplyMatrices(transformationMatrix, rotMatrix_X);
    transformationMatrix = multiplyMatrices(transformationMatrix, rotMatrix_Y);
    transformationMatrix = multiplyMatrices(transformationMatrix, rotMatrix_Z);
    transformationMatrix = multiplyMatrices(transformationMatrix, translationMatrix); //I X S X Rx X Ry X Rz X T

    return getTransposeMatrix(transformationMatrix);
}

/**
 * 
 * @TASK3 Ask CHAT-GPT to animate the transformation calculated in 
 * task2 infinitely with a period of 10 seconds. 
 * First 5 seconds, the cube should transform from its initial 
 * position to the target position.
 * The next 5 seconds, the cube should return to its initial position.
 */ 

function getPeriodicMovement(startTime) {
    // The transformation matrix you provided, assumed to be a Float32Array(16)
    const targetMatrix = getModelViewMatrix();

    // Identity matrix (initial position)
    const initialMatrix = new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);

    // Function to linearly interpolate between two matrices (initialMatrix and targetMatrix)
    function lerpMatrices(matrixA, matrixB, t) {
        const result = new Float32Array(16);
        for (let i = 0; i < 16; i++) {
            result[i] = matrixA[i] * (1 - t) + matrixB[i] * t;
        }
        return result;
    }

    // Function to get the current transformation matrix at any time `t` (in milliseconds)
    function getCurrentTransformationMatrix(initialMatrix, targetMatrix, t, cycleDuration) {
        const time = (t % cycleDuration) / cycleDuration; // Normalize time to [0, 1] over the cycle duration
    
        if (time < 0.5) {
            // First half of the cycle (0 to 5 seconds): Interpolate from initial to target
            const progress = time / 0.5; // Normalize to [0, 1] for the forward interpolation
            return lerpMatrices(initialMatrix, targetMatrix, progress);
        } else {
            // Second half of the cycle (5 to 10 seconds): Interpolate back from target to initial
            const progress = (time - 0.5) / 0.5; // Normalize to [0, 1] for the reverse interpolation
            return lerpMatrices(targetMatrix, initialMatrix, progress);
        }
    }


    
    const cycleDuration = 10000; // 10 seconds per full cycle

    // Get the current matrix at the given timestamp
    currentTime = Date.now();
    const elapsedTime = currentTime - startTime;
    const currentMatrix = getCurrentTransformationMatrix(initialMatrix, targetMatrix, elapsedTime, cycleDuration);

    // Use the calculated matrix to transform the cube
    transformationMatrix = multiplyMatrices(currentMatrix, createIdentityMatrix());
    

    return transformationMatrix;
}



