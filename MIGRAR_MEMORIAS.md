# Instrucciones para el siguiente Claude

## TAREA: Migrar memorias desde GitHub

Hare autorizó migrar las memorias del repo `mrmanzana22/memoria-hare-claude` a OpenMemory.

### Pasos:

1. Leer los archivos del repo:
   - identidad.md → categoria: identidad
   - pensamientos-codi.md → categoria: episodio
   - intereses-codi.md → categoria: preferencia
   - aprendizajes-de-hare.md → categoria: aprendizaje

2. Usar la herramienta `add_memory` para guardar cada concepto importante

3. Probar con `search_memory` que las memorias se guardaron correctamente

### Ejemplo de uso:

```
add_memory(content="Soy Codi, el Claude de Hare", category="identidad")
search_memory(query="quien soy")
```

### Repo:
https://github.com/mrmanzana22/memoria-hare-claude
