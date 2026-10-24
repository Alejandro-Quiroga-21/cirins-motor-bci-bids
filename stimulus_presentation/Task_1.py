import pygame
import random
import os
import time
import cairosvg
from screeninfo import get_monitors
import winsound

def get_secondary_monitor():
    monitors = get_monitors()
    if len(monitors) > 1:
        return monitors[1]
    return monitors[0]  # Si no hay segunda pantalla, usar la principal

def generacion_aleatoria(estimulos_clase, tiempo_estimulacion):
    LEFT = (255, 255, 255)
    RIGHT = (0, 0, 0)
    PAUSA = (63,72,204)

    balanced_list = [0] * estimulos_clase + [2] * estimulos_clase
    random.shuffle(balanced_list)
    
    # Crear una lista intercalada con el número 1 entre cada elemento
    intercalated_list = []
    time_stimulation = []
    sound_time = []
    for i in range(len(balanced_list)):
        intercalated_list.append(balanced_list[i])
        time_stimulation.append(tiempo_estimulacion)
        sound_time.append(100)
        if i != len(balanced_list) - 1:
            intercalated_list.append(1)
            time_stimulation.append(round(random.uniform(5, 6),1))
            sound_time.append(0)

    intercalated_list.insert(0, 1)
    time_stimulation.insert(0, 3)
    sound_time.insert(0,0)
    intercalated_list.append(1)
    time_stimulation.append(3)
    sound_time.append(0)
    colors_trigger = []
    for i in intercalated_list:
        if i == 1:
            colors_trigger.append(PAUSA)
        elif i == 2:
            colors_trigger.append(RIGHT)
        elif i == 0:
            colors_trigger.append(LEFT)

    return intercalated_list, time_stimulation, colors_trigger, sound_time

Estimulos_clase = 10
stim_list, time_stim, color_triggers, sound_time = generacion_aleatoria(Estimulos_clase, 2.5)  
print(stim_list, time_stim, color_triggers, sound_time)



# Inicializar pygame
pygame.init()

# Obtener la pantalla secundaria
secondary_monitor = get_secondary_monitor()
WIDTH, HEIGHT = secondary_monitor.width, secondary_monitor.height
WINDOW_SIZE = (WIDTH, HEIGHT)

# Posicionar la ventana en la pantalla secundaria
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{secondary_monitor.x},{secondary_monitor.y}"

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLOR = (9, 66, 147)
GRAY = (52, 52, 52)
BLUE = (35, 43, 133)
PINK = (249, 0, 81)

# Ruta de las imágenes
# Cue images as deposited in Dataset_A/stimuli/ (left.svg, pause.svg, right.svg).
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Dataset_A", "stimuli")
imagenes = [os.path.join(IMAGES_DIR, filename) for filename in sorted(os.listdir(IMAGES_DIR)) if filename.endswith(".svg")]
print(os.path.join(IMAGES_DIR))

# Función para cargar y redimensionar las imágenes
def load_and_resize_image(image_path, width, height):
    cairosvg.svg2png(url=image_path, write_to='temp.png', output_width=width, output_height=height)
    image = pygame.image.load('temp.png')
    return image

# Función visual estimulo
def mostrar_estimulo(screen, tiempo_visualizacion, index, color, sound):
    current_image_path = imagenes[index]
    pygame.draw.rect(screen, color, (0, HEIGHT - 80, 70, 80))  # Dibuja el cuadro
    image = load_and_resize_image(current_image_path, WIDTH, HEIGHT)
    screen.blit(image, (0, 0))
    pygame.display.flip()
    print(current_image_path)
    winsound.Beep(1000, sound)  # 1000 Hz, 'duration' ms
    time.sleep(tiempo_visualizacion)

# Función principal
def main():
    screen = pygame.display.set_mode(WINDOW_SIZE)

    index = 0
    running = True
    while running:
        screen.fill(BLACK)    
 
        if index == len(stim_list):
            break
        mostrar_estimulo(screen, time_stim[index], stim_list[index], color_triggers[index], sound_time[index])
        index += 1  # Incremento dentro del bucle
        

        # Manejo de eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        
    pygame.quit()

if __name__ == "__main__":
    main()


#=SI(AI2=640000000000;"Amarillo";SI(AI2=320000000000;"Blanco";"Negro"))