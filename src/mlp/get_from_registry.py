def get_from_registry(registry, name, category) -> list | str: 
    if name is None:
        raise ValueError(f"Le nom de {category} ne peut pas être None.")
    if not isinstance(name, str) and not isinstance(name, list):
        raise ValueError(f"Le nom de {category} doit être une chaîne de caractères ou une liste de chaînes de caractères.")
    if isinstance(name, list):
        for n in name:
            if not isinstance(n, str):
                raise ValueError(f"Le nom de {category} doit être une chaîne de caractères ou une liste de chaînes de caractères.")
            if n not in registry:
                valid = ", ".join(k for k in registry if k)
                raise ValueError(
                    f"{category} inconnue : {n!r}. "
                    f"Valeurs disponibles : {valid}"
                )
        return [registry[n] for n in name]


    if name not in registry:
        valid = ", ".join(k for k in registry if k)
        raise ValueError(
            f"{category} inconnue : {name!r}. "
            f"Valeurs disponibles : {valid}"
        )

    return registry[name]
