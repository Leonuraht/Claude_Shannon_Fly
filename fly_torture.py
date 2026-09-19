pip install ursina torch numpy

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from google.colab import output
from IPython.display import HTML, display, JSON


USE_REAL_HEMIBRAIN = True 
EDGES_CSV = "/content/traced-total-connections.csv"
NODES_CSV = "/content/traced-neurons.csv"


class UselessBoxConnectome(nn.Module):
    def __init__(self, use_real_data, edges_path, nodes_path):
        super().__init__()
        self.decay_rate = 0.85
        self.threshold = 1.0
        
        if use_real_data and os.path.exists(edges_path) and os.path.exists(nodes_path):
            edges = pd.read_csv(edges_path)
            nodes = pd.read_csv(nodes_path)
            unique_neurons = pd.concat([edges['bodyId_pre'], edges['bodyId_post']]).unique()
            self.num_neurons = len(unique_neurons)
            self.id_to_idx = {body_id: idx for idx, body_id in enumerate(unique_neurons)}
            
            nt_map = {}
            if 'nt_type' in nodes.columns:
                nt_map = nodes.set_index('bodyId')['nt_type'].to_dict()
            else:
                for nid in unique_neurons:
                    if np.random.rand() < 0.30: nt_map[nid] = 'gaba'
            
            pre_indices, post_indices, weights = [], [], []
            pre_ids = edges['bodyId_pre'].values
            post_ids = edges['bodyId_post'].values
            raw_weights = edges['weight'].values
            
            for i in range(len(edges)):
                pre_id = pre_ids[i]
                pre_indices.append(self.id_to_idx[pre_id])
                post_indices.append(self.id_to_idx[post_ids[i]])
                w = raw_weights[i] * 0.005 
                nt = str(nt_map.get(pre_id, '')).lower()
                if 'gaba' in nt or 'glut' in nt: w *= -1.0 
                weights.append(w)
            
            indices = torch.tensor([pre_indices, post_indices], dtype=torch.long)
            values = torch.tensor(weights, dtype=torch.float32)

            top_hubs = edges['bodyId_post'].value_counts().head(10).index.tolist()
            self.pain_idx = [self.id_to_idx[top_hubs[0]], self.id_to_idx[top_hubs[1]]]
            self.motor_idx = [self.id_to_idx[top_hubs[2]], self.id_to_idx[top_hubs[3]]]
            
        else:
            self.num_neurons = 130000
            num_connections = 5000000 
            indices = torch.randint(0, self.num_neurons, (2, num_connections))
            values = (torch.rand(num_connections) - 0.3) * 0.1 
            self.pain_idx = [0, 1]
            self.motor_idx = [2, 3]

        self.weights = torch.sparse_coo_tensor(indices, values, (self.num_neurons, self.num_neurons)).coalesce()
        self.membrane_potentials = torch.zeros(self.num_neurons)
        self.spikes = torch.zeros(self.num_neurons)

    def forward(self, external_stimulus):
        self.membrane_potentials *= self.decay_rate
        spikes_2d = self.spikes.unsqueeze(1) 
        internal_input = torch.sparse.mm(self.weights, spikes_2d).squeeze()
        self.membrane_potentials += internal_input + external_stimulus
        self.spikes = (self.membrane_potentials >= self.threshold).float()
        self.membrane_potentials[self.spikes > 0] = 0.0
        return self.spikes

brain = UselessBoxConnectome(USE_REAL_HEMIBRAIN, EDGES_CSV, NODES_CSV)

fly_pos = np.array([0.0, 1.0, 0.0])
switch_pos = np.array([25.0, 1.0, 25.0]) 
fly_rotation = 0.0 
fly_speed = 2.0 
fly_alive = False
death_count = 0


def compute_tick(is_switch_on):
    global fly_pos, fly_rotation, fly_alive, death_count
    

    if is_switch_on and not fly_alive:
        fly_alive = True
        fly_pos = np.array([0.0, 1.0, 0.0])
        
    stimulus = torch.zeros(brain.num_neurons)
    switch_flipped_by_fly = False

    if fly_alive:
        if is_switch_on:
            stimulus[brain.pain_idx] = 1.5  
        else:
            stimulus += (torch.rand(brain.num_neurons) * 0.02)  

        spikes = brain(stimulus)
        motor_activity = spikes[brain.motor_idx].sum().item()

        if is_switch_on and motor_activity > 0:
            direction = switch_pos - fly_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance
                ideal_rotation = np.arctan2(direction[0], direction[2])
                fly_rotation = ideal_rotation + np.random.uniform(-0.8, 0.8)
                
            forward = np.array([np.sin(fly_rotation), 0, np.cos(fly_rotation)])
            fly_pos += forward * fly_speed
            

            if distance < 1.5: 
                switch_flipped_by_fly = True
                fly_alive = False
                death_count += 1
                
        else:
            fly_rotation += np.random.uniform(-0.8, 0.8)
            forward = np.array([np.sin(fly_rotation), 0, np.cos(fly_rotation)])
            fly_pos += forward * (fly_speed * 0.4)
            
        if fly_pos[0] <= -38 or fly_pos[0] >= 38 or fly_pos[2] <= -38 or fly_pos[2] >= 38:
            fly_rotation += np.pi + np.random.uniform(-0.5, 0.5)

        fly_pos[0] = np.clip(fly_pos[0], -40, 40)
        fly_pos[2] = np.clip(fly_pos[2], -40, 40)

    return JSON({
        "x": float(fly_pos[0]), 
        "y": float(fly_pos[1]), 
        "z": float(fly_pos[2]),
        "rotation": float(fly_rotation),
        "alive": fly_alive,
        "switch_flipped": switch_flipped_by_fly,
        "death_count": death_count
    })

output.register_callback('compute_tick', compute_tick)


html_code = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
        body { margin: 0; overflow: hidden; font-family: 'Courier New', Courier, monospace; background: #000; }
        #canvas-container { width: 100%; height: 500px; position: relative; }
        
        /* SLEEK CENTERED BOTTOM UI */
        #ui-container { 
            position: absolute; 
            bottom: 20px; 
            left: 50%; 
            transform: translateX(-50%);
            z-index: 100; 
            pointer-events: none; 
            display: flex; 
            flex-direction: column; 
            align-items: center;
            gap: 12px; 
            width: 380px;
            background: rgba(20, 20, 30, 0.85);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #444;
            box-shadow: 0px 10px 30px rgba(0,0,0,0.8);
            backdrop-filter: blur(4px);
        }
        
        #useless-switch { 
            color: white; 
            border: 2px solid #fff; 
            padding: 15px 30px; 
            font-size: 22px; 
            border-radius: 8px; 
            cursor: pointer; 
            user-select: none; 
            font-weight: bold; 
            pointer-events: auto; 
            transition: 0.2s; 
            background: #555; 
            font-family: inherit; 
            width: 100%;
            letter-spacing: 1px;
        }
        #useless-switch:active { transform: scale(0.95); }
        
        #counter { color: #ffaa00; font-weight: bold; font-size: 16px; text-align: center; }
        #status { color: #00ff00; font-weight: bold; font-size: 13px; text-align: center; margin-top: 5px; }
    </style>
</head>
<body>
    <div id="canvas-container">
        <div id="ui-container">
            <button id="useless-switch">SYSTEM [OFF]</button>
            <div id="counter">EMPLOYEES SACRIFICED: 0</div>
            <div id="status">Log: System idling. Awaiting user input.</div>
        </div>
    </div>
    <script>
        let isSwitchOn = false;
        const toggleBtn = document.getElementById('useless-switch');
        const statusText = document.getElementById('status');
        const counterText = document.getElementById('counter');
        
        function updateBtnUI() {
            if (isSwitchOn) {
                toggleBtn.innerText = "SYSTEM [ON]"; 
                toggleBtn.style.background = "#ff2222"; 
                statusText.innerText = "Log: Employee spawned.Motivation: Agony ";
                statusText.style.color = '#ff4444';
                statusText.style.borderColor = '#ff4444';
            } else {
                toggleBtn.innerText = "SYSTEM [OFF]"; 
                toggleBtn.style.background = "#44aa44"; 
            }
        }

        toggleBtn.addEventListener('click', () => { 
            if (!isSwitchOn) {
                isSwitchOn = true; 
                updateBtnUI(); 
            }
        });

        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e); 
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / 500, 0.1, 1000);
        camera.position.set(0, 40, 50); 
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, 500);
        container.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        
        const light = new THREE.DirectionalLight(0xffffff, 1.2);
        light.position.set(20, 50, 20);
        scene.add(light);
        scene.add(new THREE.AmbientLight(0x404040)); 

        const floorGeo = new THREE.PlaneGeometry(100, 100);
        const floorMat = new THREE.MeshStandardMaterial({ color: 0x333333 }); 
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        scene.add(floor);
        scene.add(new THREE.GridHelper(100, 100, 0x555555, 0x222222));

        // The Fly
        const flyGroup = new THREE.Group();
        const bodyGeo = new THREE.SphereGeometry(0.5, 16, 16);
        const bodyMat = new THREE.MeshStandardMaterial({ color: 0x111111 });
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        body.scale.set(0.8, 0.8, 1.8); 
        flyGroup.add(body);
        
        const eyeGeo = new THREE.SphereGeometry(0.2, 16, 16);
        const eyeMat = new THREE.MeshStandardMaterial({ color: 0xff0000 });
        const eyeL = new THREE.Mesh(eyeGeo, eyeMat); eyeL.position.set(0.3, 0.3, 0.6);
        const eyeR = new THREE.Mesh(eyeGeo, eyeMat); eyeR.position.set(-0.3, 0.3, 0.6);
        flyGroup.add(eyeL); flyGroup.add(eyeR);
        
        const wingGeo = new THREE.PlaneGeometry(2.0, 0.8);
        const wingMat = new THREE.MeshStandardMaterial({ color: 0xeeeeee, transparent: true, opacity: 0.6, side: THREE.DoubleSide });
        const wingL = new THREE.Mesh(wingGeo, wingMat); wingL.position.set(1.0, 0.4, 0); wingL.rotation.x = Math.PI / 2;
        const wingR = new THREE.Mesh(wingGeo, wingMat); wingR.position.set(-1.0, 0.4, 0); wingR.rotation.x = Math.PI / 2;
        flyGroup.add(wingL); flyGroup.add(wingR);
        
        flyGroup.visible = false;
        scene.add(flyGroup);

        // The Physical Switch
        const switchGeo = new THREE.BoxGeometry(4, 1, 4);
        const switchMat = new THREE.MeshStandardMaterial({ color: 0x888888 });
        const physicalSwitch = new THREE.Mesh(switchGeo, switchMat);
        physicalSwitch.position.set(25, 0.5, 25);
        scene.add(physicalSwitch);
        
        const buttonTopGeo = new THREE.CylinderGeometry(1.5, 1.5, 0.5, 32);
        const buttonTopMat = new THREE.MeshStandardMaterial({ color: 0x44aa44 });
        const buttonTop = new THREE.Mesh(buttonTopGeo, buttonTopMat);
        buttonTop.position.set(25, 1.0, 25);
        scene.add(buttonTop);

        // Ghost Array for comical deaths
        const ghosts = [];
        const ghostGeo = new THREE.SphereGeometry(0.6, 16, 16);
        
        function spawnGhost(pos) {
            const ghostMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
            const ghost = new THREE.Mesh(ghostGeo, ghostMat);
            ghost.position.copy(pos);
            scene.add(ghost);
            ghosts.push(ghost);
        }

        let targetPos = new THREE.Vector3(0, 1, 0);
        let targetRot = 0;
        let wingAngle = 0;
        
        async function updateSimulation() {
            if (typeof google !== 'undefined' && google.colab) {
                const result = await google.colab.kernel.invokeFunction('compute_tick', [isSwitchOn], {});
                let state = result.data['application/json'];
                
                if (state) {
                    flyGroup.visible = state.alive;
                    targetPos.set(state.x, state.y, state.z);
                    targetRot = state.rotation;
                    counterText.innerText = "EMPLOYEES SACRIFICED: " + state.death_count;

                    if (state.switch_flipped && isSwitchOn) { 
                        isSwitchOn = false; 
                        updateBtnUI(); 
                        spawnGhost(flyGroup.position); // Poof!
                        statusText.innerText = "Log: Task completed. Employee terminated.";
                        statusText.style.color = '#00ff00';
                        statusText.style.borderColor = '#00ff00';
                    }

                    if (isSwitchOn) {
                        buttonTopMat.color.setHex(0xff0000); 
                    } else {
                        buttonTopMat.color.setHex(0x44aa44); 
                    }
                }
            }
            setTimeout(updateSimulation, 50);
        }

        updateBtnUI(); 
        updateSimulation(); 

        function animate() {
            requestAnimationFrame(animate);
            if (flyGroup.visible) {
                flyGroup.position.lerp(targetPos, 0.3); 
                let rotDiff = targetRot - flyGroup.rotation.y;
                while (rotDiff > Math.PI) rotDiff -= Math.PI * 2;
                while (rotDiff < -Math.PI) rotDiff += Math.PI * 2;
                flyGroup.rotation.y += rotDiff * 0.2;
                wingAngle += 1.8; 
                wingL.rotation.y = Math.sin(wingAngle) * 0.6;
                wingR.rotation.y = -Math.sin(wingAngle) * 0.6;
            }
            
            // Animate ghosts floating up to heaven
            for (let i = ghosts.length - 1; i >= 0; i--) {
                let ghost = ghosts[i];
                ghost.position.y += 0.15;
                ghost.rotation.y += 0.1;
                ghost.material.opacity -= 0.015;
                if (ghost.material.opacity <= 0) {
                    scene.remove(ghost);
                    ghosts.splice(i, 1);
                }
            }
            
            controls.update(); renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
"""
display(HTML(html_code))
